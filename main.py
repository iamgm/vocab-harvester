import os
import timeit
import time
import subprocess
import concurrent.futures
import threading
import argparse
from collections import deque
from string import punctuation
import glob
import shutil
# import requests
# -------------------------------------------------------------------------------
import numpy as np
import pandas as pd
# -------------------------------------------------------------------------------

# Thread-safe logging to avoid interleaved output from multiple threads
PRINT_LOCK = threading.Lock()

def thread_safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)
import nltk
from nltk.corpus import stopwords, brown, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# -------------------------------------------------------------------------------
# Optional imports - handle missing dependencies gracefully
try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    thread_safe_print("Warning: googletrans not available. Google Translate functionality disabled.")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    thread_safe_print("Warning: gTTS not available. Google TTS functionality disabled.")

try:
    from reverso_client_patched import Client as ReversoClient
    REVERSO_AVAILABLE = True
except ImportError:
    REVERSO_AVAILABLE = False
    thread_safe_print("Warning: reverso_context_api not available. Reverso translation disabled.")

try:
    from text_to_speech import Text_to_speech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    thread_safe_print("Warning: text_to_speech not available. TTS functionality disabled.")
# -------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
# File paths
COCA_WORDS_PATH = './files/wordFrequency.xlsx'
# Use absolute path to avoid shell resolution issues on Windows
FFMPEG_PATH = os.path.join(os.getcwd(), 'ffmpeg', 'ffmpeg.exe')
TEMP_DIR = './temp'
AUDIO_DIR = './audio'
COCA_CACHE_PATH = './files/coca_top_lemmas.txt'

# NLTK settings
BROWN_WORDS_COUNT = 10000

# Audio processing settings
AUDIO_SAMPLE_RATE = '24k'
AUDIO_MODELS = {
    'en': 'hfc_female',
    'ru': 'irina'
}

# Translation settings
DEFAULT_TRANSLATION_LANG = 'ru'
MAX_TRANSLATIONS_PER_WORD = 3

# Threading settings
DEFAULT_WORKER_COUNT = 4
THREAD_DELAY = 0.1

# -----------------------------------------------------------------------------
# classes
# -----------------------------------------------------------------------------
# Performance improvements implemented:
# 1. Thread-local storage for WordTranslator instances to avoid creating new objects
# 2. Pre-created translator pool for better resource management
# 3. Proper cleanup of temporary files
# 4. Renamed WordDequeue to WordProcessor for better semantics
# 5. Separated initialization from execution - added run() method
# 6. Better separation of concerns - constructor only sets up, run() executes
# 7. Thread-safe access to results dictionary with proper locking
# 8. Protection against race conditions when multiple threads write to shared data
# 9. Optimized dictionary access - use local variables instead of re-reading from dictionary
# 10. Windows-safe ffmpeg invocation: absolute path + subprocess list args (no shell)
# 11. Added `-y` to ffmpeg commands to avoid interactive overwrite prompts
# 12. Thread-safe logging via `thread_safe_print` to prevent interleaved output
# 13. Reverso retries with exponential backoff and Google Translate fallback
# 14. Per-book subfolder in `audio/` based on source filename; passed to WordProcessor


class WordExtractor:
    """
    The WordExtractor class takes the path to a book, a set of stop words, a
    set of common words, and a string of punctuation as input. It has a method
    extract_words that extracts unknown words from the book and returns them as
    a set.
    """

    def __init__(self, book_path):
        self.book_path = book_path
        # top 5000 of Top 60,000 "lemmas" from COCA corpus
        self.coca_path = COCA_WORDS_PATH
        self.coca_cache_path = COCA_CACHE_PATH
        # self.download_nltk_resources()
        self.brown_words = self.get_most_frequent_brownWords(BROWN_WORDS_COUNT)
        self._book_lemmas = None  # Initialize as None, will be set by extract_words
        # Preload COCA lemmas (with caching)
        self.coca_lemmas_set = self.load_coca_lemmas_cached(self.coca_path, self.coca_cache_path)

    def extract_words(self) -> set:
        # Extract words from the book
        # Drop punctuation, stop words, and common words
        # Return the set of unknown words
        
        # Load and process the  
        book_str = self.load_book_file(self.book_path)
        book_tokens = self.tokenize(book_str)
        book_tagged_tokens = self.pos_tag(book_tokens)
        book_lemmas = self.lemmatize(book_tagged_tokens)
        book_lemmas = self.drop_stop_words(book_lemmas)
        book_lemmas = self.drop_punctuation(book_lemmas)
        book_lemmas = self.drop_coca_lemmas(book_lemmas, self.coca_path)
        book_lemmas = self.drop_short_words(book_lemmas)
        
        # Store the result and return it
        self._book_lemmas = book_lemmas
        return book_lemmas

    def download_nltk_resources(self) -> None:
        resources = ['brown', 'punkt', 'stopwords',
                     'averaged_perceptron_tagger', 'wordnet']

        for resource in resources:
            try:
                nltk.data.find(resource)
            except LookupError:
                thread_safe_print(f"{resource} not found. Downloading...")
                nltk.download(resource)

    def get_most_frequent_brownWords(self, num) -> set:
        fdist = nltk.FreqDist(w.lower() for w in brown.words())
        return set([word for word, _ in fdist.most_common(num)])

    def load_book_file(self, path) -> str:
        with open(path, 'r') as file:
            content = file.read()
            thread_safe_print(f'Number of characters in the file: {len(content)}')
            return content

    def tokenize(self, text) -> set:
        words_set = set(word_tokenize(text.lower()))  # 2174
        thread_safe_print(f'File words_set length {len(words_set)}')
        return words_set

    def pos_tag(self, words) -> set:
        return nltk.pos_tag(words)

    def lemmatize(self, tagged_words) -> set:
        # pos map to convert NLTK pos tags to word net lemmatizer pos tags
        pos_map = {
            'J': wordnet.ADJ, 'JJ': wordnet.ADJ,
            'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,
            'N': wordnet.NOUN, 'NN': wordnet.NOUN, 'NNS': wordnet.NOUN,
            'NNP': wordnet.NOUN, 'NNPS': wordnet.NOUN,
            'V': wordnet.VERB, 'VB': wordnet.VERB, 'VBD': wordnet.VERB,
            'VBG': wordnet.VERB, 'VBN': wordnet.VERB,
            'VBP': wordnet.VERB, 'VBZ': wordnet.VERB,
            'R': wordnet.ADV, 'RB': wordnet.ADV,
            'RBR': wordnet.ADV, 'RBS': wordnet.ADV
        }

        # Perform lemmatization for each word in the set
        lemmatizer = WordNetLemmatizer()
        lemma_set = set()
        for word, pos in tagged_words:
            # Lemmatize the word based on its part of speech (POS)
            mapped_pos = pos_map.get(pos[0], wordnet.NOUN)
            lemma = lemmatizer.lemmatize(word, pos=mapped_pos)
            lemma_set.add(lemma)
        thread_safe_print(f'Lemmas set length {len(lemma_set)}')
        return lemma_set

    def load_worksheet(self, path, sheet) -> pd.DataFrame:
        return pd.read_excel(path, sheet_name=sheet)

    def drop_stop_words(self, words, lang='english') -> set:
        filtered = words - set(stopwords.words(lang))
        thread_safe_print(f'After dropping stop words {len(filtered)}')
        return filtered

    def drop_punctuation(self, words) -> set:
        filtered = set([word for word in words if word.isalpha()])
        thread_safe_print(f'After dropping punctuation {len(filtered)}')
        return filtered

    def drop_coca_lemmas(self, words, path, num=-1) -> set:
        # Use cached COCA lemmas set to avoid hitting Excel every run
        return words - self.coca_lemmas_set

    def drop_short_words(self, words) -> set:
        # drop words that are shorter than 3 chars
        filtered = {word for word in words if len(word) > 2}
        thread_safe_print(f'After dropping short words (1 or 2 chars) {len(filtered)}')
        return filtered
    
    def load_coca_lemmas_cached(self, xlsx_path: str, cache_path: str, sheet: str = '1 lemmas', num: int = 5050) -> set:
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = [line.strip() for line in f if line.strip()]
                if cached and len(cached) == num:
                    thread_safe_print(f"Loaded {len(cached)} COCA lemmas from cache: {cache_path}")
                    return set(cached)
                else:
                    thread_safe_print(f"COCA cache size mismatch ({len(cached)} != {num}), regenerating")
        except Exception as e:
            thread_safe_print(f"Warning: failed reading COCA cache '{cache_path}': {e}")

        # Fallback: read from Excel and persist cache
        try:
            df = self.load_worksheet(xlsx_path, sheet)
            if 'lemma' not in df.columns:
                raise KeyError("Expected column 'lemma' not found in COCA sheet")
            series = df['lemma'][:num] if num and num > 0 else df['lemma']
            # Ensure exactly `num` unique, non-empty lemmas in order
            ordered = [s for s in series.dropna().astype(str).str.strip().tolist() if s]
            ordered = ordered[:num]
            lemmas = set(ordered)
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    for lemma in ordered:
                        f.write(lemma + "\n")
                thread_safe_print(f"Saved {len(ordered)} COCA lemmas to cache: {cache_path}")
            except Exception as e:
                thread_safe_print(f"Warning: failed writing COCA cache '{cache_path}': {e}")
            return lemmas
        except Exception as e:
            thread_safe_print(f"Warning: failed loading COCA from Excel '{xlsx_path}': {e}")
            return set()
    
    @property
    def book_lemmas(self) -> set:
        if self._book_lemmas is None:
            return self.extract_words()
        return self._book_lemmas

    @book_lemmas.setter
    def book_lemmas(self, val) -> None:
        self._book_lemmas = val


class WordTranslator:
    """
    The WordTranslator class. It has methods to get word translations using,
    create text-to-speech audio files for the word and its
    translations, concatenate audio files for the words, and create back
    translations audio files for self testing.
    """

    def __init__(self):
        # Initialize optional components based on availability
        if GOOGLETRANS_AVAILABLE:
            self.translator = Translator()
        else:
            self.translator = None
            
        if REVERSO_AVAILABLE:
            self.ctx_reverso = ReversoClient("en", "ru")
        else:
            self.ctx_reverso = None
            
        if TTS_AVAILABLE:
            self.TTS = Text_to_speech()
        else:
            self.TTS = None


    def get_translations(self, word: str, dest: str = "ru"):
        # Prefer Reverso, but gracefully fall back to Google if blocked or failing
        if REVERSO_AVAILABLE:
            res = self.get_translations_reverso(word, dest)
            # If Reverso failed (returns only original word), try Google as fallback
            if res and (len(res) > 1 or (len(res) == 1 and res[0] != word)):
                return res
            if GOOGLETRANS_AVAILABLE:
                g_res = self.get_translations_googletrans(word, dest)
                return g_res if g_res else (word,)
            return (word,)
        elif GOOGLETRANS_AVAILABLE:
            return self.get_translations_googletrans(word, dest)
        else:
            thread_safe_print(f"Warning: No translation service available for word '{word}'")
            return (word,)  # Return the original word as fallback
    
    # context reverso implementation
    def get_translations_reverso(self, word: str, dest: str = "ru") -> tuple:
        if not REVERSO_AVAILABLE:
            thread_safe_print("Warning: Reverso translation service not available")
            return (word,)
        # Retry with exponential backoff to mitigate 403/temporary blocks
        max_attempts = 3
        backoff_seconds = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                res = tuple(self.ctx_reverso.get_translations(word))
                return res
            except Exception as e:
                err = str(e)
                thread_safe_print(f"Error with Reverso translation (attempt {attempt}/{max_attempts}): {err}")
                if attempt == max_attempts:
                    break
                # Longer backoff if looks like a 403/ban
                sleep_time = backoff_seconds * (2 if '403' in err or 'Forbidden' in err else 1)
                time.sleep(sleep_time)
                backoff_seconds *= 2
        return (word,)
    
    # googletrans implementation
    def get_translations_googletrans(self, word: str, dest: str = "ru") -> tuple:
        if not GOOGLETRANS_AVAILABLE:
            thread_safe_print("Warning: Google Translate service not available")
            return (word,)
        try:
            result = self.translator.translate(word, dest=dest)
            more_translations = result.extra_data.get('more_translations', [])
            return (result.text, *(more_translations))
        except Exception as e:
            thread_safe_print(f"Error with Google Translate: {e}")
            return (word,)

    # NOT USED
    # gTTS implementation doesn't work (gTTSError: 429 (Too Many Requests))
    def create_audio_file_gtts(self, word: str, fn: str, dir: str = "./temp",
                          lang="en", slow=False) -> None:
        if not GTTS_AVAILABLE:
            print("Warning: gTTS not available, skipping audio creation")
            return
        try:
            # Create text-to-speech audio file for the word
            tts = gTTS(word, lang=lang, slow=slow)
            tts.save(f'{dir}/{fn}.mp3')
        except Exception as e:
            print(f"Error creating gTTS audio: {e}")

    def create_audio_file(self, word: str, fn: str, _dir: str = TEMP_DIR,
                          lang="en", slow=False) -> None:
        # Create text-to-speech audio file for the word
        if not TTS_AVAILABLE:
            print("Warning: TTS not available, skipping audio creation")
            return
            
        try:
            md = AUDIO_MODELS.get(lang, AUDIO_MODELS['en'])
            self.TTS.speak(word, md, lang, fn, _dir)
            inp_wav = os.path.normpath(os.path.join(_dir, f"{fn}.wav"))
            out_mp3 = os.path.normpath(os.path.join(_dir, f"{fn}.mp3"))
            cmd = [
                FFMPEG_PATH,
                '-y',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', inp_wav,
                '-ar', AUDIO_SAMPLE_RATE,
                out_mp3,
            ]
            thread_safe_print("Running:", ' '.join(cmd))
            subprocess.run(cmd, check=True)
        except Exception as e:
            thread_safe_print(f"Error creating TTS audio: {e}")

    def concatenate_audio_files(self, word: str, dir: str = AUDIO_DIR) -> None:
        # Concatenate audio files the word and translations
        mp3_files = glob.glob(f"{TEMP_DIR}/{word}_*.mp3")
        with open(os.path.join(TEMP_DIR, f"{word}.txt"), "w") as f:
            f.write(f"file '{word}.mp3' \n")
            for file in mp3_files:
                f.write(f"file '{os.path.basename(file)}' \n")
        concat_list = os.path.normpath(os.path.join(TEMP_DIR, f"{word}.txt"))
        out_path = os.path.normpath(os.path.join(dir, f"{word}.mp3"))
        cmd = [
            FFMPEG_PATH,
            '-y',
            '-hide_banner',
            '-loglevel', 'error',
            '-safe', '0',
            '-f', 'concat',
            '-i', concat_list,
            '-c', 'copy',
            out_path,
        ]
        thread_safe_print("Running concatenation:", ' '.join(cmd))
        subprocess.run(cmd, check=True)

    def create_back_translations_audio_files(self, words: list) -> None:
        # Create back translations audio files for self testing
        pass


class WordProcessor:
    """
    The WordProcessor class takes a set of words as input. It has a method
    to_dequeue that converts the set of words to a dequeue. It also has a
    method pop_words that pops words from the dequeue using ThreadPoolExecutor
    and runs the WordTranslator class for each word.
    """

    def __init__(self, words: set, num_workers: int = None, output_dir: str = AUDIO_DIR):
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

        # Ensure output directory exists
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.words_dequeue = deque(list(words))
        self.words_dict = {}
        
        # Thread-local storage for WordTranslator instances
        self.thread_local = threading.local()
        
        # Pre-create translators for better performance
        self.num_workers = num_workers if num_workers is not None else DEFAULT_WORKER_COUNT
        self.translators_pool = [WordTranslator() for _ in range(self.num_workers)]
        self.translator_index = 0
        self.translator_lock = threading.Lock()
        
        # Lock for thread-safe access to results dictionary
        self.results_lock = threading.Lock()

        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        # print(self.words_dict)

    def run(self):
        """Start processing words with the configured number of workers"""
        self.pop_words(self.num_workers)
        return self.words_dict

    def get_results(self):
        """Get the processed words dictionary in a thread-safe manner"""
        with self.results_lock:
            return self.words_dict.copy()  # Return a copy to avoid external modifications

    def get_processing_stats(self):
        """Get processing statistics in a thread-safe manner"""
        with self.results_lock:
            return {
                'total_words_processed': len(self.words_dict),
                'words_with_translations': len([w for w in self.words_dict.values() if len(w) > 1]),
                'total_translations': sum(len(trs) for trs in self.words_dict.values())
            }

    def pop_word(self, index):
        # collections.deque is thread-safe for atomic operations like popleft()
        # No lock needed for this operation
        word = self.words_dequeue.popleft()
        # thread_safe_print(f"word {word}", end=' ')

        # self.mod_path = Path(__file__).parent
        cwd = os.getcwd()
        self.temp_folder = os.path.join(cwd, TEMP_DIR)
        
        # Проверяем, есть ли уже переводчик для этого потока
        if not hasattr(self.thread_local, 'translator'):
            # Если нет, получаем переводчик из пула
            with self.translator_lock:
                translator_index = self.translator_index % len(self.translators_pool)
                self.translator_index += 1
                self.thread_local.translator = self.translators_pool[translator_index]
        
        trslr = self.thread_local.translator  # Используем один и тот же объект
        # thread_safe_print(word)
        # trs = trslr.get_translations(word)

        # self.words_dict[word] = trs
        
        # trslr.create_audio_file(word, word, self.temp_folder, "en", slow=True) 
                    
        try:
            thread_safe_print(word)
            trs = trslr.get_translations(word)
            
            # Thread-safe write to results dictionary
            with self.results_lock:
                self.words_dict[word] = trs
            
            trslr.create_audio_file(word, word,  self.temp_folder, "en") 
            thread_safe_print(*trs)
            
            # Use local variable trs directly - no need to read from dictionary again
            for i, tr in enumerate(trs):
                if i < MAX_TRANSLATIONS_PER_WORD:
                    # thread_safe_print(f"   {i}.{tr}")
                    trslr.create_audio_file(
                        tr, f"{word}_{i}",  self.temp_folder, lang="ru")
            trslr.concatenate_audio_files(word, dir=self.output_dir)
            # thread_safe_print(f"{str(index).zfill(3)}. {word}: {self.words_dict[word]}")
        except Exception as e:
            thread_safe_print(f"{word} : {e}")
            # pass

        time.sleep(THREAD_DELAY)

        # thread_safe_print(f"{str(index).zfill(3)}. {word}")

    def cleanup(self):
        """Clean up resources and temporary files"""
        try:
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
                thread_safe_print(f"Cleaned up temporary directory: {TEMP_DIR}")
        except Exception as e:
            thread_safe_print(f"Error during cleanup: {e}")

    def pop_words(self, num_workers: int):
        # Create a semaphore to limit the number of pending tasks
        semaphore = threading.Semaphore(num_workers)
        
        total_tasks = len(self.words_dequeue)
        thread_safe_print(f"Starting processing of {total_tasks} words with {num_workers} workers")
        
        # Создаем пул ОДИН РАЗ для всех задач
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Создаем список всех задач
            futures = []
            
            for i in range(total_tasks):
                # Обертываем задачу для семафора
                def task_wrapper(index=i):
                    semaphore.acquire()
                    try:
                        return self.pop_word(index)
                    finally:
                        semaphore.release()
                
                # Отправляем задачу в уже существующий пул
                futures.append(executor.submit(task_wrapper))
            
            # Логируем начальное состояние пула
            thread_safe_print(f"Thread pool created with {num_workers} workers")
            thread_safe_print(f"Submitted {len(futures)} tasks to the pool")

            # Ждем завершения всех задач
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Можно обработать результат или ошибку
                    completed += 1
                    if completed % 10 == 0:  # Логируем каждые 10 завершенных задач
                        thread_safe_print(f"Progress: {completed}/{total_tasks} tasks completed")
                except Exception as e:
                    thread_safe_print(f"Task failed with error: {e}")
            
            thread_safe_print(f"All tasks completed. Successfully processed {completed}/{total_tasks} words.")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Extract unknown words from text files')
    parser.add_argument('--book-path', '-b', 
                       default='./txt/The Warren Buffett Pilot Story - The Importance of Making a NOT To Do List.txt',
                       help='Path to the text file to process')
    parser.add_argument('--workers', '-w', type=int, default=DEFAULT_WORKER_COUNT,
                       help=f'Number of worker threads (default: {DEFAULT_WORKER_COUNT})')
    parser.add_argument('--output-dir', '-o', default=AUDIO_DIR,
                       help=f'Base output directory for audio files (default: {AUDIO_DIR})')
    
    args = parser.parse_args()
    
    # Create extractor and process the book
    # If input is a .vtt, copy/convert it to a .txt in TEMP_DIR and use that
    source_path = args.book_path
    if source_path.lower().endswith('.vtt'):
        os.makedirs(TEMP_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(source_path))[0]
        txt_path = os.path.join(TEMP_DIR, base + '.txt')
        try:
            with open(source_path, 'r', encoding='utf-8', errors='ignore') as src, \
                 open(txt_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            thread_safe_print(f"Converted VTT to TXT: {txt_path}")
            source_path = txt_path
        except Exception as e:
            thread_safe_print(f"Failed to convert VTT to TXT, using original: {e}")

    extractor = WordExtractor(source_path)
    thread_safe_print(f"Processing book: {source_path}")
    thread_safe_print(f"Number of unknown words found: {len(extractor.book_lemmas)}")
    
    # Get the words and process them
    words = list(extractor.book_lemmas)
    thread_safe_print(f"Words: {words}")
    
    # Derive a per-book audio subfolder name from the source file
    book_base = os.path.splitext(os.path.basename(source_path))[0]
    safe_book_base = ''.join(c for c in book_base if c.isalnum() or c in (' ', '_', '-', '.'))
    per_book_dir = os.path.join(args.output_dir, safe_book_base)
    if not os.path.exists(per_book_dir):
        os.makedirs(per_book_dir)

    # Process words with specified number of workers
    thread_safe_print(f"Using {args.workers} worker threads")
    words_processor = WordProcessor(words, num_workers=args.workers, output_dir=per_book_dir)
    
    # Start processing
    results = words_processor.run()
    
    # Get processing statistics
    stats = words_processor.get_processing_stats()
    
    # Clean up temporary files
    words_processor.cleanup()
    
    thread_safe_print(f"Processing completed. Audio files saved to: {per_book_dir}")
    thread_safe_print(f"Successfully processed {stats['total_words_processed']} words")
    thread_safe_print(f"Words with translations: {stats['words_with_translations']}")
    thread_safe_print(f"Total translations generated: {stats['total_translations']}")

if __name__ == "__main__":
    main()