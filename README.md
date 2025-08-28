# Vocab Harvester - Unknown Words Application

A multi-threaded application for processing text files, extracting unknown words, translating them, and generating audio pronunciations.

## Features

- **Multi-threaded processing** with configurable worker threads
- **Automatic word extraction** from text files
- **Translation support** via Google Translate and Reverso
- **Text-to-speech generation** for pronunciation
- **Audio processing** with FFmpeg
- **Cross-platform compatibility**

## Installation

### Prerequisites

- Python 3.7+
- Poetry (Python package manager)
- FFmpeg (download from [ffmpeg.org](https://ffmpeg.org/download.html))

### Setup with Poetry

```bash
# Clone the repository
git clone https://github.com/iamgm/vocab-harvester.git
cd vocab-harvester

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# NLTK data will be downloaded automatically on first run
# (The script handles this internally via download_nltk_resources())

# Download FFmpeg and place ffmpeg.exe in the ffmpeg/ directory
# (or add FFmpeg to your system PATH)

# Verify installation
python -c "import numpy, pandas, nltk; print('Dependencies installed successfully!')"
```

### Alternative: Manual Installation

```bash
# Install required packages
pip install numpy pandas nltk googletrans==4.0.0rc1 gtts

# Install optional packages for enhanced functionality
pip install text_to_speech reverso_context_api
```

## Quick Start

### Windows (простой способ)
```batch
# Basic usage with default settings
run.bat

# Process a specific book file
run.bat --book-path ./txt/my_book.txt

# Use custom number of worker threads
run.bat --workers 8

# Specify output directory
run.bat --output-dir ./my_audio

# Get help
run.bat --help
```

### Linux/Mac (стандартный способ)
```bash
# Basic usage with default settings
python main.py

# Process a specific book file
python main.py --book-path ./txt/my_book.txt

# Use custom number of worker threads
python main.py --workers 8

# Specify output directory
python main.py --output-dir ./my_audio

# Get help
python main.py --help
```

## Configuration

### Environment Variables

```bash
# File paths (optional, defaults to script location)
export UNKNOWN_WORDS_BASE_DIR=/path/to/your/application

# NLTK settings
export BROWN_WORDS_COUNT=10000

# Audio processing
export AUDIO_SAMPLE_RATE=24k
export AUDIO_MODEL_EN=hfc_female
export AUDIO_MODEL_RU=irina

# Translation settings
export DEFAULT_TRANSLATION_LANG=ru
export MAX_TRANSLATIONS_PER_WORD=3

# Threading
export DEFAULT_WORKER_COUNT=4
export THREAD_DELAY=0.1

# Text processing
export MIN_WORD_LENGTH=3
export COCA_TOP_WORDS=-1

# File encoding
export DEFAULT_FILE_ENCODING=utf-8

# Logging
export LOG_LEVEL=INFO
```

### Command Line Arguments

```bash
# Specify a different book file
python main.py --book-path ./txt/my_book.txt
python main.py -b ./txt/my_book.txt

# Specify number of worker threads
python main.py --workers 8

# Specify output directory
python main.py --output-dir ./my_audio
python main.py -o ./my_audio

# Get help
python main.py --help
```

## Architecture & Performance

### Multi-threading Implementation

The application uses a **single ThreadPoolExecutor** for all tasks, avoiding the performance issues of creating new thread pools for each word:

```python
# Single thread pool for all tasks
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(task_wrapper) for i in range(total_tasks)]
```

**Key improvements:**
- **Single thread pool**: Created once and reused for all tasks
- **Semaphore management**: Proper resource control with guaranteed cleanup
- **Progress monitoring**: Real-time tracking of completed tasks
- **Error handling**: Robust error handling at task level
- **Scalable workers**: Easy configuration of thread count

### Performance Benefits

1. **Eliminates thread creation/destruction overhead**
2. **Optimal resource utilization**
3. **Better scalability with configurable worker count**
4. **Real-time progress monitoring**
5. **Proper semaphore management**

### Example Performance Output

```
Starting processing of 150 words with 8 workers
Thread pool created with 8 workers
Submitted 150 tasks to the pool
Progress: 10/150 tasks completed
Progress: 20/150 tasks completed
...
All tasks completed. Successfully processed 150/150 words.
```

## File Structure

```
vocab-harvester/
├── main.py                 # Main application
├── text_to_speech.py      # TTS functionality
├── reverso_client_patched.py  # Reverso translation client
├── ffmpeg/                # FFmpeg binaries
│   └── ffmpeg.exe
├── data/                  # Data files
│   ├── wordFrequency.xlsx
│   ├── coca_top_lemmas.txt
│   └── words.txt
├── temp/                  # Temporary files
├── audio/                 # Generated audio files
└── txt/                   # Input text files
    └── your_text_file.txt
```

## Dependencies

### Required
- Python 3.7+
- NLTK
- pandas
- numpy
- FFmpeg

### Optional (for enhanced functionality)
- googletrans (Google Translate)
- gtts (Google Text-to-Speech)
- text_to_speech (local TTS)

## Usage Examples

### Windows (простой способ)
```batch
# Process default book with 4 threads
run.bat

# Process specific book with 8 threads
run.bat -b ./txt/my_book.txt -w 8

# Custom output directory with 16 threads
run.bat -b ./txt/my_book.txt -w 16 -o ./custom_audio

# Process with specific settings
run.bat --book-path ./txt/my_book.txt --workers 12 --output-dir ./my_audio
```

### Linux/Mac (стандартный способ)
```bash
# Process default book with 4 threads
python main.py

# Process specific book with 8 threads
python main.py -b ./txt/my_book.txt -w 8

# Custom output directory with 16 threads
python main.py -b ./txt/my_book.txt -w 16 -o ./custom_audio

# Process with specific settings
python main.py --book-path ./txt/my_book.txt --workers 12 --output-dir ./my_audio
```

## Configuration Details

### Default Constants (from main.py)

```python
# File paths
COCA_WORDS_PATH = './files/wordFrequency.xlsx'
FFMPEG_PATH = os.path.join(os.getcwd(), 'ffmpeg', 'ffmpeg.exe')
TEMP_DIR = './temp'
AUDIO_DIR = './audio'

# NLTK settings
BROWN_WORDS_COUNT = 10000

# Audio processing
AUDIO_SAMPLE_RATE = '24k'
AUDIO_MODELS = {'en': 'hfc_female', 'ru': 'irina'}

# Translation settings
DEFAULT_TRANSLATION_LANG = 'ru'
MAX_TRANSLATIONS_PER_WORD = 3

# Threading settings
DEFAULT_WORKER_COUNT = 4
THREAD_DELAY = 0.1
```

## Portability Features

- **No hardcoded paths** - All paths are configurable
- **Cross-platform compatibility** - Uses `os.path.join()` for path construction
- **Environment-based configuration** - Easy setup without code changes
- **Command line flexibility** - Runtime configuration options

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure `ffmpeg.exe` is in the `ffmpeg/` directory
2. **Missing dependencies**: Install optional packages for enhanced functionality
3. **File encoding issues**: Use `DEFAULT_FILE_ENCODING` environment variable
4. **Thread performance**: Adjust `DEFAULT_WORKER_COUNT` based on your system

### Performance Tuning

- **Increase workers** for CPU-intensive tasks: `--workers 16`
- **Adjust thread delay** if needed: `export THREAD_DELAY=0.05`
- **Monitor progress** to identify bottlenecks
- **Use appropriate audio models** for your language needs

## Development Notes

The application has been refactored to:
- Eliminate hardcoded paths
- Implement efficient multi-threading
- Provide flexible configuration options
- Ensure cross-platform compatibility
- Optimize resource usage and performance
