import os
import time
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import openai
import concurrent.futures
from tqdm import tqdm
import argparse
from pathlib import Path
import mobi
import tempfile
import zipfile
import shutil

# Configure OpenAI API key - set your API key as an environment variable or enter it here
openai.api_key = os.environ.get("OPENAI_API_KEY", "your_api_key_here")

# Configuration
CONFIG = {
    "voice": "alloy",  # OpenAI voice options: alloy, echo, fable, onyx, nova, shimmer
    "model": "tts-1",  # OpenAI TTS model
    "output_dir": "audiobook_output",
    "max_workers": 5,  # Number of parallel workers for conversion
    "chunk_size": 4000,  # Character limit per audio segment
}

def extract_text_from_epub(epub_path):
    """Extract text content from an ePub file."""
    book = epub.read_epub(epub_path)
    chapters = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text().strip()
            if text:  # Only add non-empty chapters
                chapters.append(text)
    
    return chapters

def extract_text_from_mobi(mobi_path):
    """Extract text content from a MOBI file."""
    # Create a temporary directory for extraction
    temp_dir = tempfile.mkdtemp()
    try:
        # Extract MOBI to temporary directory
        tempname = os.path.join(temp_dir, "temp.mobi")
        shutil.copy(mobi_path, tempname)
        
        # Use the mobi library to extract content
        m = mobi.Mobi(tempname)
        m.parse()
        
        # Get the text
        text = m.book_html.decode('utf-8', errors='ignore')
        soup = BeautifulSoup(text, 'html.parser')
        chapters = []
        
        # Try to find chapter divisions or just use the whole text
        chapter_tags = soup.find_all(['h1', 'h2', 'h3'])
        if chapter_tags:
            current_text = ""
            for tag in soup.body.contents:
                if tag in chapter_tags and current_text:
                    chapters.append(current_text.strip())
                    current_text = str(tag)
                else:
                    if hasattr(tag, 'get_text'):
                        current_text += tag.get_text()
                    else:
                        current_text += str(tag)
            if current_text:
                chapters.append(current_text.strip())
        else:
            chapters.append(soup.get_text().strip())
        
        return chapters
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def chunk_text(text, max_chars=CONFIG["chunk_size"]):
    """Split text into chunks that don't exceed the maximum character limit."""
    chunks = []
    current_chunk = ""
    
    # Split by sentences to avoid cutting in the middle of a sentence
    sentences = text.replace('. ', '.\n').replace('! ', '!\n').replace('? ', '?\n').split('\n')
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def text_to_speech(text, output_path):
    """Convert text to speech using OpenAI API and save to output_path."""
    try:
        response = openai.audio.speech.create(
            model=CONFIG["model"],
            voice=CONFIG["voice"],
            input=text
        )
        response.stream_to_file(output_path)
        return True
    except Exception as e:
        print(f"Error during TTS conversion: {e}")
        return False

def process_chunk(args):
    """Process a single text chunk to audio (for parallel processing)."""
    chunk_idx, chunk, base_output_dir, filename_prefix = args
    output_file = os.path.join(base_output_dir, f"{filename_prefix}_{chunk_idx:04d}.mp3")
    success = text_to_speech(chunk, output_file)
    return success, chunk_idx, len(chunk)

def convert_ebook_to_audiobook(ebook_path):
    """Convert an ebook file to an audiobook."""
    # Create output directory
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the filename without extension to use as a prefix
    filename_prefix = Path(ebook_path).stem
    book_output_dir = os.path.join(output_dir, filename_prefix)
    os.makedirs(book_output_dir, exist_ok=True)
    
    print(f"Converting {ebook_path} to audiobook...")
    start_time = time.time()
    
    # Extract text based on file type
    file_ext = Path(ebook_path).suffix.lower()
    if file_ext == '.epub':
        chapters = extract_text_from_epub(ebook_path)
    elif file_ext == '.mobi':
        chapters = extract_text_from_mobi(ebook_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    print(f"Extracted {len(chapters)} chapters.")
    
    # Process all chapters into chunks
    all_chunks = []
    for chapter_idx, chapter in enumerate(chapters):
        chunks = chunk_text(chapter)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append((
                len(all_chunks),  # Global chunk index
                chunk,
                book_output_dir, 
                filename_prefix
            ))
    
    # Process chunks in parallel with progress bar
    total_chars = sum(len(chunk[1]) for chunk in all_chunks)
    processed_chars = 0
    
    print(f"Total text length: {total_chars} characters")
    print(f"Converting to speech using {CONFIG['max_workers']} parallel workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = [executor.submit(process_chunk, chunk_args) for chunk_args in all_chunks]
        
        with tqdm(total=total_chars, unit='chars') as pbar:
            for future in concurrent.futures.as_completed(futures):
                success, chunk_idx, chunk_len = future.result()
                processed_chars += chunk_len
                pbar.update(chunk_len)
                remaining = total_chars - processed_chars
                pbar.set_description(f"Processed: {processed_chars}/{total_chars} chars (Remaining: {remaining} chars)")
    
    duration = time.time() - start_time
    print(f"Conversion completed in {duration:.2f} seconds")
    print(f"Audio files saved to {book_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert eBook to audiobook using OpenAI's text-to-speech API")
    parser.add_argument("ebook_path", help="Path to the eBook file (.epub, .mobi)")
    parser.add_argument("--voice", default=CONFIG["voice"], help="OpenAI voice to use")
    parser.add_argument("--model", default=CONFIG["model"], help="OpenAI TTS model to use")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"], help="Output directory for audio files")
    parser.add_argument("--workers", type=int, default=CONFIG["max_workers"], help="Number of parallel workers")
    parser.add_argument("--chunk-size", type=int, default=CONFIG["chunk_size"], help="Maximum characters per audio segment")
    
    args = parser.parse_args()
    
    # Update configuration from arguments
    CONFIG["voice"] = args.voice
    CONFIG["model"] = args.model
    CONFIG["output_dir"] = args.output_dir
    CONFIG["max_workers"] = args.workers
    CONFIG["chunk_size"] = args.chunk_size
    
    convert_ebook_to_audiobook(args.ebook_path)