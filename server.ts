import express, { Request, Response, Application, RequestHandler, NextFunction } from 'express';
import session from 'express-session';
import { print } from 'listening-on';
import { randomUUID } from 'node:crypto';
import sqlite3 from 'sqlite3';
import bcrypt from 'bcrypt';
import path from 'path';
import { main } from './collect-images';

// Extend express-session and express types
declare module 'express-session' {
  interface SessionData {
    token?: string;
    userId?: number;
  }
}

declare module 'express' {
  interface Request {
    user_id?: number;
  }
}

// Interfaces
interface User {
  id: number;
  username: string;
  password: string;
  avatar: string | null;
  email: string | null;
  created_at: string;
}

interface Session {
  id: number;
  token: string;
  user_id: number;
  created_at: string;
}

interface Image {
  id: number;
  src: string;
  alt: string | null;
  filename: string | null;
  user_id: number | null;
  search_id: number | null;
  is_relevant: boolean | null;
}

interface Search {
  id: number;
  user_id: number;
  keyword: string;
  image_count: number;
  search_time: string;
}

// Initialize SQLite database
const DB_PATH = 'db.sqlite3';
const db = new sqlite3.Database(DB_PATH, (err) => {
  if (err) {
    console.error('Database connection error:', err.message);
  } else {
    console.log('Connected to SQLite database');
    db.serialize(() => {
      // Create user table
      db.run(`
        CREATE TABLE IF NOT EXISTS user (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL UNIQUE,
          password TEXT NOT NULL,
          avatar TEXT,
          email TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
      `);

      // Create session table
      db.run(`
        CREATE TABLE IF NOT EXISTS session (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          token TEXT NOT NULL UNIQUE,
          user_id INTEGER NOT NULL,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES user(id)
        )
      `);

      // Create searches table
      db.run(`
        CREATE TABLE IF NOT EXISTS searches (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          keyword TEXT NOT NULL,
          image_count INTEGER NOT NULL DEFAULT 0,
          search_time TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES user(id)
        )
      `);

      // Create images table
      db.run(`
        CREATE TABLE IF NOT EXISTS images (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          src TEXT NOT NULL,
          alt TEXT,
          filename TEXT,
          user_id INTEGER,
          search_id INTEGER,
          is_relevant BOOLEAN,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES user(id),
          FOREIGN KEY (search_id) REFERENCES searches(id)
        )
      `);
    });
  }
});

// Create Express app
const server: Application = express();

// Middleware
server.use(express.static(path.join(__dirname, 'public')));
server.use('/dog_images', express.static(path.join(__dirname, 'dog_images')));
server.use(express.urlencoded({ extended: true }));
server.use(express.json());
server.use(
  session({
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: false,
    cookie: { maxAge: 24 * 60 * 60 * 1000 },
  })
);

// Authentication middleware
const authenticate: RequestHandler = (req: Request, res: Response, next: NextFunction): void => {
  const token = req.headers.authorization || (req.query.token as string) || req.session.token;
  if (!token) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  db.get('SELECT user_id FROM session WHERE token = ?', [token], (err, session: Session | undefined) => {
    if (err) {
      console.error('Database error in authenticate:', err.message);
      res.status(500).json({ error: 'server error' });
      return;
    }
    if (!session) {
      res.status(401).json({ error: 'Invalid token' });
      return;
    }
    req.user_id = session.user_id;
    req.session.userId = session.user_id;
    next();
  });
};

// Routes
server.get('/', (req: Request, res: Response, next: NextFunction) => {
  const token = req.session.token || req.query.token || req.headers.authorization;
  if (token) {
    db.get('SELECT user_id FROM session WHERE token = ?', [token], (err, session: Session | undefined) => {
      if (!err && session) {
        res.redirect('/search');
      } else {
        res.sendFile(path.join(__dirname, 'public', 'index.html'));
      }
    });
  } else {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
  }
});

server.get('/register', (req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'public', 'register.html'));
});

server.post('/register', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  const { username, email, password } = req.body;
  if (!username || !password) {
    res.status(400).json({ error: 'Username and password are required' });
    return;
  }
  if (username.length < 3 || username.length > 32) {
    res.status(400).json({ error: 'Username must be between 3 and 32 characters' });
    return;
  }
  if (password.length < 6) {
    res.status(400).json({ error: 'Password must be at least 6 characters' });
    return;
  }

  try {
    const row = await new Promise<User | undefined>((resolve, reject) => {
      db.get('SELECT id FROM user WHERE username = ?', [username], (err, row: User | undefined) => {
        if (err) reject(err);
        else resolve(row);
      });
    });

    if (row) {
      res.status(409).json({ error: 'Username already exists' });
      return;
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    await new Promise<void>((resolve, reject) => {
      db.run(
        'INSERT INTO user (username, password, email, avatar) VALUES (?, ?, ?, ?)',
        [username, hashedPassword, email || null, null],
        (err) => {
          if (err) reject(err);
          else resolve();
        }
      );
    });
    res.redirect('/login');
  } catch (err) {
    console.error('Register error:', err);
    res.status(500).json({ error: 'server error' });
  }
});

server.get('/login', (req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'public', 'login.html'));
});

server.post('/login', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  const { username, password } = req.body;
  if (!username || !password) {
    res.status(400).json({ error: 'Username and password are required' });
    return;
  }

  try {
    const user = await new Promise<User | undefined>((resolve, reject) => {
      db.get('SELECT id, password FROM user WHERE username = ?', [username], (err, user: User | undefined) => {
        if (err) reject(err);
        else resolve(user);
      });
    });

    if (!user) {
      res.status(401).json({ error: 'Invalid credentials' });
      return;
    }

    if (await bcrypt.compare(password, user.password)) {
      const token = randomUUID();
      await new Promise<void>((resolve, reject) => {
        db.run(
          'INSERT INTO session (token, user_id) VALUES (?, ?)',
          [token, user.id],
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      });
      req.user_id = user.id;
      req.session.userId = user.id;
      req.session.token = token;
      res.json({ token });
    } else {
      res.status(401).json({ error: 'Invalid credentials' });
    }
  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({ error: 'server error' });
  }
});

server.get('/logout', (req: Request, res: Response) => {
  const token = req.session.token;
  req.session.destroy((err) => {
    if (err) console.error('Logout error:', err.message);
    if (token) {
      db.run('DELETE FROM session WHERE token = ?', [token], (err) => {
        if (err) console.error('Delete session error:', err.message);
      });
    }
    res.redirect('/login');
  });
});

server.get('/search', authenticate, (req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'public', 'search.html'));
});

server.post('/search', authenticate, async (req: Request, res: Response): Promise<void> => {
  const { keywords } = req.body;
  if (!keywords || !Array.isArray(keywords) || keywords.length === 0) {
    res.status(400).json({ error: 'At least one keyword is required' });
    return;
  }
  try {
    await main(req.user_id!, keywords);
    res.redirect('/results');
  } catch (err: any) {
    console.error('Search error:', err.message);
    res.status(500).json({ error: 'Search failed' });
  }
});

server.get('/results', authenticate, (req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'public', 'results.html'));
});

server.get('/api/images', authenticate, (req: Request, res: Response) => {
  db.all(
    'SELECT id, src, alt, filename, is_relevant FROM images WHERE user_id = ? ORDER BY created_at DESC',
    [req.user_id!],
    (err, rows) => {
      if (err) {
        console.error('Database error in /api/images:', err.message);
        return res.status(500).json({ error: 'server error' });
      }
      console.log('Images fetched:', rows.map((r: any) => ({ id: r.id, src: r.src, filename: r.filename })));
      res.json(rows);
    }
  );
});

server.get('/api/searches', authenticate, (req: Request, res: Response) => {
  db.all(
    'SELECT id, keyword, search_time, image_count FROM searches WHERE user_id = ? ORDER BY search_time DESC',
    [req.user_id!],
    (err, rows) => {
      if (err) {
        console.error('Database error in /api/searches:', err.message);
        return res.status(500).json({ error: 'server error' });
      }
      res.json(rows);
    }
  );
});

server.get('/logs', authenticate, (req: Request, res: Response) => {
  res.download(path.join(__dirname, 'log.txt'), 'image_log.txt', (err) => {
    if (err) {
      console.error('Log download error:', err.message);
      res.status(500).send('Log download failed');
    }
  });
});

// Error-handling middleware
server.use((err: any, req: Request, res: Response, next: NextFunction) => {
  console.error('Error:', err);
  res.status(500).json({ error: 'server error' });
});

// Start server
const PORT = 3000;
server.listen(PORT, () => {
  print(PORT);
});