import re
from collections import defaultdict
from datetime import datetime

def summarize_log(log_file='image_collection.log', output_file='log_summary.txt'):
    """
    Summarize the content of image_collection.log, extracting key metrics and saving to console/file.
    
    Args:
        log_file (str): Path to the log file (default: 'image_collection.log').
        output_file (str): Path to save the summary (default: 'log_summary.txt').
    
    Returns:
        dict: Summary metrics for further use.
    """
    # Initialize counters and storage
    total_images = 0
    successful_images = 0
    failed_downloads = 0
    failed_compressions = 0
    total_errors = 0
    keywords = defaultdict(int)  # Track images per keyword
    current_keyword = None
    
    # Regular expressions for parsing
    total_images_re = re.compile(r'Total images collected and processed: (\d+)')
    successful_re = re.compile(r'Successfully processed image: dog_images/.*\.jpg')
    failed_download_re = re.compile(r'(Failed to download|Error downloading) .*:')
    failed_compression_re = re.compile(r'Could not compress dog_images/temp_\d+\.jpg to under 50KB')
    error_re = re.compile(r' - ERROR - ')
    keyword_start_re = re.compile(r'Processing keyword: (.*)')
    keyword_complete_re = re.compile(r'Completed keyword (.*), total images: (\d+)')
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Extract total images
                total_match = total_images_re.search(line)
                if total_match:
                    total_images = int(total_match.group(1))
                
                # Count successful images
                if successful_re.search(line):
                    successful_images += 1
                
                # Count failed downloads
                if failed_download_re.search(line):
                    failed_downloads += 1
                
                # Count failed compressions
                if failed_compression_re.search(line):
                    failed_compressions += 1
                
                # Count errors
                if error_re.search(line):
                    total_errors += 1
                
                # Track keyword progress
                keyword_start_match = keyword_start_re.search(line)
                if keyword_start_match:
                    current_keyword = keyword_start_match.group(1)
                
                keyword_complete_match = keyword_complete_re.search(line)
                if keyword_complete_match:
                    keyword = keyword_complete_match.group(1)
                    images = int(keyword_complete_match.group(2))
                    keywords[keyword] = images
                    current_keyword = None
    
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
        return {}
    except Exception as e:
        print(f"Error reading log file: {str(e)}")
        return {}
    
    # Prepare summary
    summary = {
        'total_images': total_images,
        'successful_images': successful_images,
        'failed_downloads': failed_downloads,
        'failed_compressions': failed_compressions,
        'total_errors': total_errors,
        'keywords_processed': len(keywords),
        'keywords': dict(keywords)
    }
    
    # Format summary for console
    summary_text = [
        "Image Collection Log Summary",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 40,
        f"Total Images Processed: {total_images}",
        f"Successful Images: {successful_images}",
        f"Failed Downloads: {failed_downloads}",
        f"Failed Compressions: {failed_compressions}",
        f"Total Errors: {total_errors}",
        f"Keywords Processed: {len(keywords)}",
        "Top 5 Keywords by Image Count:"
    ]
    
    # Add top 5 keywords
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
    for keyword, count in sorted_keywords:
        summary_text.append(f"  - {keyword}: {count} images")
    
    # Add issues if target not met
    if total_images < 3000:
        summary_text.append("\nIssues:")
        summary_text.append(f"  - Only {total_images} images collected (target: 3000–5000).")
        if failed_compressions > 0:
            summary_text.append(f"  - {failed_compressions} images failed compression (check 50KB limit).")
        if failed_downloads > 0:
            summary_text.append(f"  - {failed_downloads} images failed to download (check URLs or network).")
        if total_errors > 0:
            summary_text.append(f"  - {total_errors} errors occurred (review log for details).")
    
    # Print to console
    print("\n".join(summary_text))
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_text))
        print(f"\nSummary saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving summary to '{output_file}': {str(e)}")
    
    return summary

if __name__ == "__main__":
    summarize_log()