import pygame
import sys
import time
import math
import threading
import os
import json
from pathlib import Path
import requests
import io
from dotenv import load_dotenv
import wave
import pyaudio
import tempfile
import whisper
import torch
import queue
from datetime import datetime
import numpy as np
from scipy.io import wavfile
import scipy.signal

import openai
# import anthropic

# Utility functions
def view_session_responses(session_path):
    """View responses from a completed session"""
    responses_path = os.path.join(session_path, "responses.json")
    metadata_path = os.path.join(session_path, "metadata.json")
    
    if not os.path.exists(responses_path):
        print("No responses found for this session")
        return
        
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    print(f"\n{'='*60}")
    print(f"Session: {metadata['session_id']}")
    print(f"Created: {metadata['created']}")
    print(f"Status: {metadata['status']}")
    print(f"{'='*60}\n")
    
    # Load and display responses
    with open(responses_path, 'r') as f:
        responses = json.load(f)
        
    for response in responses:
        # Convert timestamp to readable format
        timestamp = datetime.fromtimestamp(response['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"Cycle {response['cycle']} ({timestamp}):")
        print(f"{response['response']}")
        print(f"{'-'*60}\n")
        
def export_session_to_text(session_path, output_file):
    """Export session responses to a readable text file"""
    responses_path = os.path.join(session_path, "responses.json")
    metadata_path = os.path.join(session_path, "metadata.json")
    
    if not os.path.exists(responses_path):
        return False
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    with open(responses_path, 'r') as f:
        responses = json.load(f)
        
    # Write to text file
    with open(output_file, 'w') as f:
        f.write(f"EMDR Processing Session Report\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Session ID: {metadata['session_id']}\n")
        f.write(f"Date: {metadata['created']}\n")
        f.write(f"Total Cycles: {len(responses)}\n\n")
        f.write(f"{'='*60}\n\n")
        
        for response in responses:
            timestamp = datetime.fromtimestamp(response['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Cycle {response['cycle']} - {timestamp}\n")
            f.write(f"{'-'*40}\n")
            f.write(f"{response['response']}\n\n")
            
    return True

def get_existing_target_files():
    """Get a list of existing target image files"""
    files = []
    for file in os.listdir('.'):
        if file.startswith('Target_Image_') and file.endswith('.txt'):
            files.append(file)
    return sorted(files)

# ===== Linked List Implementation for Processing Responses =====
class ResponseNode:
    """Node for linked list storing processing phase responses"""
    def __init__(self, cycle_number, response_text, timestamp):
        self.cycle_number = cycle_number
        self.response_text = response_text
        self.timestamp = timestamp
        self.next = None

class ProcessingResponseList:
    """Linked list to store all processing phase responses"""
    def __init__(self):
        self.head = None
        self.tail = None
        self.count = 0
    
    def add_response(self, cycle_number, response_text):
        """Add a new response to the linked list"""
        new_node = ResponseNode(cycle_number, response_text, time.time())
        
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        
        self.count += 1
    
    def get_all_responses(self):
        """Return all responses as a list"""
        responses = []
        current = self.head
        while current:
            responses.append({
                'cycle': current.cycle_number,
                'response': current.response_text,
                'timestamp': current.timestamp
            })
            current = current.next
        return responses
    
    def save_to_file(self, filename):
        """Save all responses to a file"""
        responses = self.get_all_responses()
        try:
            with open(filename, 'w') as f:
                json.dump(responses, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving responses: {e}")
            return False

# ===== PROCESSING SESSION MANAGER =====
class ProcessingSessionManager:
    """Manages processing sessions and handles resume functionality"""
    def __init__(self):
        self.sessions_dir = "processing_sessions"
        os.makedirs(self.sessions_dir, exist_ok=True)
        
    def create_session(self):
        """Create a new processing session directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"
        session_path = os.path.join(self.sessions_dir, session_id)
        os.makedirs(session_path, exist_ok=True)
        
        # Create session metadata
        metadata = {
            "session_id": session_id,
            "created": timestamp,
            "status": "active",
            "audio_files": {}
        }
        
        metadata_path = os.path.join(session_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return session_id, session_path
        
    def save_audio_reference(self, session_path, cycle, audio_file):
        """Save reference to audio file in session metadata"""
        metadata_path = os.path.join(session_path, "metadata.json")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Move audio file to session directory
        new_filename = f"cycle_{cycle}_audio.wav"
        new_path = os.path.join(session_path, new_filename)
        os.rename(audio_file, new_path)
        
        # Update metadata
        metadata["audio_files"][str(cycle)] = {
            "filename": new_filename,
            "status": "pending",
            "created": time.time()
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return new_path
        
    def mark_audio_transcribed(self, session_path, cycle):
        """Mark an audio file as transcribed and delete it"""
        metadata_path = os.path.join(session_path, "metadata.json")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Update status
        if str(cycle) in metadata["audio_files"]:
            audio_info = metadata["audio_files"][str(cycle)]
            audio_info["status"] = "transcribed"
            audio_info["transcribed_at"] = time.time()
            
            # Delete the audio file
            audio_path = os.path.join(session_path, audio_info["filename"])
            try:
                os.remove(audio_path)
            except:
                pass
                
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def get_pending_sessions(self):
        """Get all sessions with pending audio files"""
        pending_sessions = []
        
        for session_dir in os.listdir(self.sessions_dir):
            session_path = os.path.join(self.sessions_dir, session_dir)
            metadata_path = os.path.join(session_path, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Check for pending audio files
                pending_count = sum(1 for audio in metadata["audio_files"].values() 
                                  if audio["status"] == "pending")
                
                if pending_count > 0:
                    pending_sessions.append({
                        "session_id": metadata["session_id"],
                        "session_path": session_path,
                        "pending_count": pending_count,
                        "metadata": metadata
                    })
                    
        return pending_sessions
        
    def load_session_responses(self, session_path):
        """Load existing responses from a session"""
        responses_path = os.path.join(session_path, "responses.json")
        if os.path.exists(responses_path):
            with open(responses_path, 'r') as f:
                return json.load(f)
        return []

# ===== BACKGROUND TRANSCRIPTION WORKER =====
class TranscriptionWorker:
    """Background worker thread for processing audio transcriptions"""
    def __init__(self, whisper_model, has_cuda):
        self.whisper_model = whisper_model
        self.has_cuda = has_cuda
        self.audio_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self.current_transcription = None
        self.session_manager = ProcessingSessionManager()
        
    def start(self):
        """Start the background worker thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=False)  # Not daemon anymore
        self.worker_thread.start()
        print("Background transcription worker started")
        
    def stop(self):
        """Stop the background worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()  # Wait indefinitely for completion
        print("Background transcription worker stopped")
        
    def add_audio(self, audio_file, cycle_number, session_path, processing_responses):
        """Add an audio file to the transcription queue"""
        # Save audio reference to session
        new_path = self.session_manager.save_audio_reference(session_path, cycle_number, audio_file)
        
        self.audio_queue.put({
            'file': new_path,
            'cycle': cycle_number,
            'session_path': session_path,
            'processing_responses': processing_responses,
            'timestamp': time.time()
        })
        print(f"Audio file added to transcription queue (Queue size: {self.audio_queue.qsize()})")
        
    def add_pending_audio(self, audio_path, cycle, session_path, processing_responses):
        """Add a previously saved audio file to the queue"""
        self.audio_queue.put({
            'file': audio_path,
            'cycle': cycle,
            'session_path': session_path,
            'processing_responses': processing_responses,
            'timestamp': time.time()
        })
        
    def get_status(self):
        """Get current transcription status"""
        return {
            'queue_size': self.audio_queue.qsize(),
            'current': self.current_transcription,
            'is_running': self.running and self.worker_thread.is_alive()
        }
        
    def _load_audio_without_ffmpeg(self, audio_path):
        """Load audio file using scipy"""
        try:
            # Read wav file
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32 normalized between -1 and 1
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                # Calculate the resampling ratio
                resample_ratio = 16000 / sample_rate
                new_length = int(len(audio_data) * resample_ratio)
                
                # Use scipy's resample for better quality
                audio_data = scipy.signal.resample(audio_data, new_length)
            
            return audio_data
            
        except Exception as e:
            print(f"Error loading audio file with scipy: {e}")
            return None
    
    def _process_queue(self):
        """Process audio files from the queue"""
        while self.running or not self.audio_queue.empty():  # Process until stopped AND queue empty
            try:
                # Wait for audio file with timeout
                item = self.audio_queue.get(timeout=1.0)
                
                self.current_transcription = f"Cycle {item['cycle']}"
                print(f"Processing audio file from queue (Cycle {item['cycle']})")
                print(f"Audio file path: {item['file']}")
                
                # Check if file exists
                if not os.path.exists(item['file']):
                    print(f"ERROR: Audio file not found: {item['file']}")
                    continue
                    
                start_time = time.time()
                
                # Transcribe the audio
                try:
                    print(f"Starting transcription on {'GPU' if self.has_cuda else 'CPU'}...")
                    
                    # Load audio without ffmpeg
                    audio_data = self._load_audio_without_ffmpeg(item['file'])
                    
                    if audio_data is None:
                        print("Failed to load audio file")
                        continue
                    
                    # Transcribe using the loaded audio data
                    result = self.whisper_model.transcribe(
                        audio_data,
                        language="en",
                        fp16=self.has_cuda,
                        verbose=False
                    )
                    
                    transcription = result["text"].strip()
                    processing_time = time.time() - start_time
                    
                    print(f"Transcription complete in {processing_time:.1f}s: {transcription[:50]}...")
                    
                    # Add to linked list
                    item['processing_responses'].add_response(item['cycle'], transcription)
                    
                    # Save updated responses to file
                    responses_path = os.path.join(item['session_path'], "responses.json")
                    item['processing_responses'].save_to_file(responses_path)
                    print(f"Saved responses to: {responses_path}")
                    
                    # Mark as transcribed and delete audio
                    self.session_manager.mark_audio_transcribed(item['session_path'], item['cycle'])
                    print(f"Marked cycle {item['cycle']} as transcribed and deleted audio file")
                    
                except Exception as e:
                    print(f"Transcription error: {e}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    self.current_transcription = None
                    
            except queue.Empty:
                # No items in queue, continue waiting
                continue
            except Exception as e:
                print(f"Worker thread error: {e}")
                import traceback
                traceback.print_exc()

# ===== Parameters  =====
# API Keys 
load_dotenv()
OPENAI_API_KEY = os.getenv('OPEN_AI_API_EMDR_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_EMDR_KEY_5k')
# ANTHROPIC_API_KEY = ""  

# ElevenLabs settings
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Visual parameters
CIRCLE_COLOR = (255, 0, 0)          # Red
CIRCLE_DIAMETER_CM = 2.0    
BACKGROUND_COLOR = (80, 80, 80)     # Dark grey currently
MARGIN_CM = 1.0                     # from screen edge

# Animation parameters
OSCILLATIONS_PER_SECOND = 1.2  # 1.2Hz from minor online research
OSCILLATION_DURATION = 3.0    # Cycle duration - Ask therapist

# Session parameters
TOTAL_CYCLES = 3

# Text parameters
PROMPT_TEXT = "What did you notice?"
CONTINUE_TEXT = "(tap spacebar to begin recording)"
RECORDING_TEXT = "Recording... (tap spacebar to stop)"
TRANSCRIBING_TEXT = "Transcribing your response..."
FEEDBACK_TEXT = "Notice that"
TEXT_COLOR = (255, 255, 255)  # White 
FONT_SIZE = 36
SMALL_FONT_SIZE = 24
MENU_FONT_SIZE = 48

FEEDBACK_DISPLAY_TIME = 2.0  # How long "Notice that" is shown in seconds
FADE_DURATION = 1.0          # Length of fade effect

# Audio recording parameters
RECORDING_RATE = 16000
RECORDING_CHANNELS = 1
RECORDING_CHUNK = 1024
RECORDING_FORMAT = pyaudio.paInt16

# Menu parameters
MENU_OPTION_COLOR = (200, 200, 200)  # Light grey 
MENU_HIGHLIGHT_COLOR = (255, 255, 0)  # Highlighted option - yellow

# File paths
AUDIO_DIR = "audio_files"
RECORDINGS_DIR = "recordings"
PROCESSING_RESPONSES_DIR = "processing_responses"
TARGET_IMAGES_DIR = "target_images"
QUESTION_AUDIO_FILES = [
    "question_1.mp3",  # "Can you describe what you saw?"
    "question_2.mp3",  # "Are there any sounds?"
    "question_3.mp3",  # "Are there any smells?"
    "question_4.mp3",  # "Do you feel any physical sensations?"
    "question_5.mp3"  # "And right now pay attention to your body, what do you feel and where?"
]
WHAT_NOTICED_AUDIO = "what_noticed.mp3"
CUE_IN_AUDIO = "cue_in_script.mp3"

# Target identification questions
TARGET_QUESTIONS = [
    "Can you describe what you see?",
    "Are there any sounds?", 
    "Are there any smells?",
    "Do you feel any physical sensations?",
    "And right now pay attention to your body, what do you feel and where?"
]

TARGET_QUESTION_KEYS = [
    "visual",
    "auditory", 
    "smell",
    "physical",
    "body"
]

# LLM Prompt for cue-in script generation
CUE_IN_PROMPT = """Transform the following trauma memory responses into a gentle, second-person cue-in script that helps the user recall the memory for EMDR therapy processing. 

Start with: "With your eyes closed or a soft gaze in front of you, take a deep breath, and remember the..."

Then convert each response from first person to second person present tense, creating a flowing narrative that guides them back to the traumatic moment. Use gentle, therapeutic language.

Do not add or invent any additional details, use only the information in the user's responses.

Example transformation:
- "I remember the smell of smoke" → "You remember the smell of smoke"
- "I felt pain in my chest" → "You feel the pain in your chest"
- "I saw the car coming" → "You see the car coming toward you"

User's responses:
Visual: {visual}
Auditory: {auditory}
Smell: {smell}
Physical: {physical}
Body: {body}


Create a cohesive, therapeutic cue-in script:"""

# ===== AUDIO RECORDING HANDLER CLASS =====
class AudioRecorder:
    def __init__(self, whisper_model_size="auto"):
        """
        Initialize the audio recorder with local Whisper
        
        Args:
            whisper_model_size: Size of Whisper model - "auto", "tiny", "base", "small", "medium", "large"
                               "auto" = automatically choose based on GPU availability
                               tiny = 39M parameters, fastest, least accurate
                               base = 74M parameters, good balance
                               small = 244M parameters, better accuracy
                               medium = 769M parameters, even better accuracy
                               large = 1550M parameters, best accuracy but slowest
        """
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.recording_thread = None
        
        # Create directories if they don't exist
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        os.makedirs(PROCESSING_RESPONSES_DIR, exist_ok=True)
        
        # Check for GPU availability
        try:
            import torch
            self.has_cuda = torch.cuda.is_available()
            if self.has_cuda:
                print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
                print("Whisper will use GPU for faster transcription")
            else:
                print("No NVIDIA GPU detected. Whisper will use CPU")
        except ImportError:
            print("PyTorch not installed. Installing with Whisper...")
            self.has_cuda = False
        
        # Auto select model based on hardware
        if whisper_model_size == "auto":
            if self.has_cuda:
                # GPU model
                whisper_model_size = "small"
                print(f"GPU detected: Using {whisper_model_size} model")
            else:
                # CPU needs better model to ensure accuracy
                whisper_model_size = "medium"
                print(f"CPU only: Using {whisper_model_size} model")
        
        # Load local Whisper model
        try:
            import whisper
            print(f"Loading Whisper model ({whisper_model_size})...")
            print("This may take a moment on first run...")
            
            # Load model with appropriate device
            self.whisper_model = whisper.load_model(
                whisper_model_size,
                device="cuda" if self.has_cuda else "cpu"
            )
            
            print(f"Whisper model loaded successfully on {'GPU' if self.has_cuda else 'CPU'}!")
            
        except ImportError:
            print("\nWhisper not installed. Please install it with:")
            print("pip install openai-whisper")
            print("\nFor GPU support, install PyTorch with CUDA:")
            print("Visit https://pytorch.org/get-started/locally/")
            raise ImportError("Please install openai-whisper package")
        
    #just putting this here one sec
    # ok so putting this here now fixes the issue where responses
    # during target image identification werent being transcribed
    # to target image files
    def _load_audio_without_ffmpeg(self, audio_path):
        """Load audio file using scipy"""
        try:
            # Read wav file
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32 normalized between -1 and 1
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                # Calculate the resampling ratio
                resample_ratio = 16000 / sample_rate
                new_length = int(len(audio_data) * resample_ratio)
                
                # Use scipy's resample for better quality
                audio_data = scipy.signal.resample(audio_data, new_length)
            
            return audio_data
            
        except Exception as e:
            print(f"Error loading audio file with scipy: {e}")
            return None
    
        
    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return
            
        self.frames = []
        self.is_recording = True
        
        # Open audio stream
        self.stream = self.pyaudio.open(
            format=RECORDING_FORMAT,
            channels=RECORDING_CHANNELS,
            rate=RECORDING_RATE,
            input=True,
            frames_per_buffer=RECORDING_CHUNK
        )
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        
    def _record(self):
        """Record audio in a separate thread"""
        while self.is_recording:
            try:
                data = self.stream.read(RECORDING_CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"Recording error: {e}")
                break
                
    def stop_recording(self):
        """Stop recording and save the audio file"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join()
            
        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        # Save recording to a temporary file
        if self.frames:
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                dir=RECORDINGS_DIR,
                delete=False
            )
            
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(RECORDING_CHANNELS)
                wf.setsampwidth(self.pyaudio.get_sample_size(RECORDING_FORMAT))
                wf.setframerate(RECORDING_RATE)
                wf.writeframes(b''.join(self.frames))
                
            return temp_file.name
            
        return None
        
    def transcribe_audio(self, audio_file):
        """Transcribe audio using local Whisper model"""
        try:
            print(f"Transcribing on {'GPU' if self.has_cuda else 'CPU'}...")

            # Chekc if file exists
            if not os.path.exists(audio_file):
                print(f"ERROR: Audio file not found: {audio_file}")
                return ""
            
            audio_data = self._load_audio_without_ffmpeg(audio_file)

            if audio_data is None:
                print("Failed to load audio file with scipy")
                return ""
            
            
            # Transcribe with local model
            result = self.whisper_model.transcribe(
                audio_data,
                language="en",
                fp16=self.has_cuda,  # Use FP16 on GPU
                verbose=False
            )
            
            transcription = result["text"].strip()
            print(f"Transcription complete: {transcription[:50]}...")
            
            return transcription
            
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return ""
            
    def cleanup(self):
        """Clean up resources"""
        if self.stream:
            self.stream.close()
        self.pyaudio.terminate()

# ===== AUDIO HANDLER CLASS =====
class AudioHandler:
    def __init__(self):
        """Initialize the audio handler"""
        # using pygame mixer for audio playback
        pygame.mixer.init()
        
        # Create audio directory if it doesn't exist
        os.makedirs(AUDIO_DIR, exist_ok=True)
        
        # Track current audio playback
        self.current_sound = None
        self.is_playing = False
        
    def generate_question_audio_files(self):
        """Generate audio files for the 5 target identification questions"""
        if not ELEVENLABS_API_KEY:
            print("ElevenLabs API key not set. Cannot generate audio files.")
            return False
            
        print("Generating question audio files...")
        
        for i, question in enumerate(TARGET_QUESTIONS):
            filename = os.path.join(AUDIO_DIR, QUESTION_AUDIO_FILES[i])
            
            # Skip if file already exists
            if os.path.exists(filename):
                print(f"Audio file {filename} already exists, skipping...")
                continue
                
            if self.generate_elevenlabs_audio(question, filename):
                print(f"Generated: {filename}")
            else:
                print(f"Failed to generate: {filename}")
                return False
                
        # Generate "What did you notice?" audio
        notice_filename = os.path.join(AUDIO_DIR, WHAT_NOTICED_AUDIO)
        if not os.path.exists(notice_filename):
            if self.generate_elevenlabs_audio("What did you notice?", notice_filename):
                print(f"Generated: {notice_filename}")
            else:
                print(f"Failed to generate: {notice_filename}")
                return False
                
        print("All question audio files generated successfully!")
        return True
        
    def generate_elevenlabs_audio(self, text, filename):
        """Generate audio using ElevenLabs API"""
        try:
            # ElevenLabs API endpoint
            url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{ELEVENLABS_VOICE_ID}"
            
            # Request headers
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            
            # Request data
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            # Make API request
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # Save audio file
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error generating audio: {e}")
            return False
            
    def play_audio_file(self, filename):
        """Play an audio file"""
        try:
            filepath = os.path.join(AUDIO_DIR, filename)
            if not os.path.exists(filepath):
                print(f"Audio file not found: {filepath}")
                return False
                
            # Load and play the sound
            self.current_sound = pygame.mixer.Sound(filepath)
            self.current_sound.play()
            self.is_playing = True
            
            return True
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
            
    def is_audio_playing(self):
        """Check if audio is currently playing"""
        if self.current_sound:
            return pygame.mixer.get_busy()
        return False
        
    def stop_audio(self):
        """Stop current audio playback"""
        pygame.mixer.stop()
        self.is_playing = False

# ===== LLM HANDLER CLASS =====
class LLMHandler:
    def __init__(self):
        """Initialize the LLM handler"""
        # Initialize OpenAI client
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            
        # Initialize Anthropic client (commented out for now)
        # if ANTHROPIC_API_KEY:
        #     self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
    def generate_cue_in_script(self, responses):
        """Generate a cue-in script from target image responses"""
        try:
            # Format the responses into the prompt
            prompt = CUE_IN_PROMPT.format(
                visual=responses[0] if len(responses) > 0 else "No visual response",
                auditory=responses[1] if len(responses) > 1 else "No auditory response",
                smell=responses[2] if len(responses) > 2 else "No smell response",
                physical=responses[3] if len(responses) > 3 else "No physical response",
                body=responses[4] if len(responses) > 4 else "No body response"
            )
            
            # Use OpenAI API
            if OPENAI_API_KEY:
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a skilled EMDR therapist creating cue-in scripts."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
            
            # Anthropic API (commented out for now)
            # if ANTHROPIC_API_KEY:
            #     response = self.anthropic_client.messages.create(
            #         model="claude-3-sonnet-20240229",
            #         max_tokens=500,
            #         messages=[
            #             {"role": "user", "content": prompt}
            #         ]
            #     )
            #     return response.content[0].text.strip()
            
            return "No API key configured for LLM."
            
        except Exception as e:
            print(f"Error generating cue-in script: {e}")
            return "Error generating cue-in script."

# ===== HELPER FUNCTIONS =====
def cm_to_pixels(cm, dpi=96):
    """Convert centimeters to pixels based on screen DPI"""
    # Standard assumption: 96 DPI (dots per inch)
    # 1 inch = 2.54 cm, so 1 cm = 96/2.54 pixels
    return int(cm * dpi / 2.54)

def get_screen_info():
    """Get screen dimensions and calculate usable area"""
    # Get the current screen resolution
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    
    # Calculate margins in pixels
    margin_pixels = cm_to_pixels(MARGIN_CM)
    
    # Calculate usable area for circle movement
    usable_width = screen_width - (2 * margin_pixels)
    usable_height = screen_height - (2 * margin_pixels)
    
    return screen_width, screen_height, margin_pixels, usable_width, usable_height

def calculate_circle_position(elapsed_time):
    """Calculate the x position of the circle based on elapsed time"""
    # Calculate how far through the oscillation cycle we are
    cycle_progress = (elapsed_time * OSCILLATIONS_PER_SECOND) % 1.0
    
    # Use sine wave to create smooth oscillation
    # sin(0) = 0, sin(π/2) = 1, sin(π) = 0, sin(3π/2) = -1, sin(2π) = 0
    angle = cycle_progress * 2 * math.pi
    
    # Convert sine wave (-1 to 1) to screen position (left margin to right margin)
    sine_value = math.sin(angle)
    
    # Map sine wave to screen coordinates
    _, _, margin_pixels, usable_width, _ = get_screen_info()
    center_x = margin_pixels + (usable_width // 2)
    amplitude = usable_width // 2
    
    x_position = center_x + (sine_value * amplitude)
    
    return int(x_position)

def get_next_target_filename():
    """Get the next available target image filename in the target_images folder"""
    # Create directory if it doesn't exist
    os.makedirs(TARGET_IMAGES_DIR, exist_ok=True)
    
    # Start with Target_Image_1.txt and increment until we find an unused name
    counter = 1
    while True:
        filename = os.path.join(TARGET_IMAGES_DIR, f"Target_Image_{counter}.txt")
        if not os.path.exists(filename):
            return filename
        counter += 1

def get_existing_target_files():
    """Get a list of existing target image files from the target_images folder"""
    os.makedirs(TARGET_IMAGES_DIR, exist_ok=True)

    files = []
    for file in os.listdir(TARGET_IMAGES_DIR):
        if file.startswith('Target_Image_') and file.endswith('.txt'):
            files.append(file)
    return sorted(files)

def save_target_responses(responses):
    """Save target image responses to a file in the target_images folder"""
    filename = get_next_target_filename()
    
    try:
        # Extract just the filename for display purposes
        display_name = os.path.basename(filename)
        counter = display_name.split('_')[2].split('.')[0]
        
        with open(filename, 'w') as f:
            f.write(f"Target Image {counter}\n")
            f.write("=" * 50 + "\n\n")
            
            # Write timestamp
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write each response with its label
            labels = ["Visual", "Auditory", "Smell", "Physical", "Body"]
            for i, (label, response) in enumerate(zip(labels, responses)):
                f.write(f"{label}: {response}\n\n")
                
        print(f"Target image saved to {filename}")
        return filename
        
    except Exception as e:
        print(f"Error saving target image: {e}")
        return None
    
def load_target_responses(filename):
    """Load target image responses from a file"""
    try:
        # If filename doesn't include the full path, construct it
        if not filename.startswith(TARGET_IMAGES_DIR):
            filepath = os.path.join(TARGET_IMAGES_DIR, filename)
        else:
            filepath = filename
            
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse the responses
        responses = []
        labels = ["Visual:", "Auditory:", "Smell:", "Physical:", "Body:"]
        
        for i, label in enumerate(labels):
            start = content.find(label)
            if start != -1:
                start += len(label)
                
                # Find the end (next label or end of file)
                if i < len(labels) - 1:
                    end = content.find(labels[i + 1])
                    if end == -1:
                        end = len(content)
                else:
                    end = len(content)
                
                response = content[start:end].strip()
                responses.append(response)
            else:
                responses.append("")
                
        return responses
        
    except Exception as e:
        print(f"Error loading target responses: {e}")
        return []

# ===== MAIN PROGRAM CLASS =====
class EMDRProgram:
    def __init__(self):
        """Initialize the EMDR program"""
        # Initialize pygame
        pygame.init()
        
        # Set up the display in fullscreen mode
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("EMDR Therapy Program")

        # Hide the mouse cursor
        pygame.mouse.set_visible(False)
        
        # Get screen dimensions and calculations
        self.screen_width, self.screen_height, self.margin_pixels, self.usable_width, self.usable_height = get_screen_info()
        
        # Calculate circle properties
        self.circle_radius = cm_to_pixels(CIRCLE_DIAMETER_CM) // 2
        self.circle_y = self.screen_height // 2  # Center vertically
        
        # Set up fonts for text display
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.small_font = pygame.font.Font(None, SMALL_FONT_SIZE)
        self.menu_font = pygame.font.Font(None, MENU_FONT_SIZE)
        
        # Initialize handlers
        self.audio_handler = AudioHandler()
        # Initialize audio recorder with auto model selection
        # Will use "tiny" on CPU for speed, "base" on GPU for quality
        self.audio_recorder = AudioRecorder(whisper_model_size="auto")
        self.llm_handler = LLMHandler()
        
        # Initialize program state - start with menu
        self.current_cycle = 0
        self.state = "menu"  #States: menu, target_identification, recording, transcribing, target_selection, cue_in_generate, cue_in_review, cue_in_audio, oscillating, waiting, processing_recording, feedback, fading
        self.start_time = time.time()
        self.feedback_start_time = 0
        self.fade_start_time = 0
        
        # Menu state variables
        self.menu_selected = 0  # 0 = Target ID, 1 = Begin Processing
        
        # Target identification state variables
        self.current_question = 0
        self.target_responses = []
        self.current_transcription = ""
        self.audio_played = False
        
        # Target selection variables
        self.target_files = []
        self.selected_target = 0
        
        # Cue-in state variables
        self.cue_in_script = ""
        self.cue_in_generated = False
        # cue_in_states Branch
        self.cue_in_script_file = None
        self.cue_in_audio_generated = False

        # Processing session variables
        self.session_manager = ProcessingSessionManager()
        self.current_session_id = None
        self.current_session_path = None
        self.processing_responses = None
        
        # Initialize background transcription worker
        self.transcription_worker = TranscriptionWorker(
            self.audio_recorder.whisper_model,
            self.audio_recorder.has_cuda
        )
        self.transcription_worker.start()
        
        # Check for pending sessions on startup
        self.check_pending_sessions()
        
        # Set up clock for frame rate control
        self.clock = pygame.time.Clock()
        
    def draw_circle(self, alpha=255):
        """Draw the oscillating circle at its current position"""
        # Calculate current position based on elapsed time
        elapsed_time = time.time() - self.start_time
        circle_x = calculate_circle_position(elapsed_time)
        
        # Create a surface for the circle with alpha transparency
        circle_surface = pygame.Surface((self.circle_radius * 2, self.circle_radius * 2), pygame.SRCALPHA)
        
        # Draw the circle on the surface with specified alpha
        color_with_alpha = (*CIRCLE_COLOR, alpha)
        pygame.draw.circle(circle_surface, color_with_alpha, (self.circle_radius, self.circle_radius), self.circle_radius)
        
        # Blit the circle surface to the main screen
        self.screen.blit(circle_surface, (circle_x - self.circle_radius, self.circle_y - self.circle_radius))
        
    def draw_text(self, text, font, color, y_position):
        """Draw centered text on the screen"""
        # Render the text
        text_surface = font.render(text, True, color)
        
        # Calculate position to center the text horizontally
        text_rect = text_surface.get_rect()
        text_rect.centerx = self.screen_width // 2
        text_rect.y = y_position
        
        # Draw the text on the screen
        self.screen.blit(text_surface, text_rect)
        
    def check_pending_sessions(self):
        """Check for and resume pending sessions"""
        pending_sessions = self.session_manager.get_pending_sessions()
        
        print(f"Checking for pending sessions... Found {len(pending_sessions)} session(s)")
        
        for session in pending_sessions:
            print(f"Found pending session: {session['session_id']} with {session['pending_count']} pending files")
            
            # Create a new ProcessingResponseList and load existing responses
            responses_list = ProcessingResponseList()
            existing_responses = self.session_manager.load_session_responses(session['session_path'])
            
            print(f"Loaded {len(existing_responses)} existing responses")
            
            # Rebuild the linked list from saved responses
            for resp in existing_responses:
                responses_list.add_response(resp['cycle'], resp['response'])
            
            # Queue pending audio files
            queued_count = 0
            for cycle, audio_info in session['metadata']['audio_files'].items():
                if audio_info['status'] == 'pending':
                    audio_path = os.path.join(session['session_path'], audio_info['filename'])
                    if os.path.exists(audio_path):
                        print(f"Queueing audio file: {audio_path}")
                        self.transcription_worker.add_pending_audio(
                            audio_path,
                            int(cycle),
                            session['session_path'],
                            responses_list
                        )
                        queued_count += 1
                    else:
                        print(f"WARNING: Audio file not found: {audio_path}")
                        
            print(f"Queued {queued_count} audio files for transcription")
                        
    def draw_menu(self):
        """Draw the main menu"""
        # Draw title
        title_y = self.screen_height // 2 - 150
        self.draw_text("EMDR Therapy Program", self.menu_font, TEXT_COLOR, title_y)
        
        # Draw menu options
        option1_y = self.screen_height // 2 - 50
        option2_y = self.screen_height // 2 + 20
        
        # Highlight selected option
        option1_color = MENU_HIGHLIGHT_COLOR if self.menu_selected == 0 else MENU_OPTION_COLOR
        option2_color = MENU_HIGHLIGHT_COLOR if self.menu_selected == 1 else MENU_OPTION_COLOR
        
        self.draw_text("1. Identify Target Image", self.font, option1_color, option1_y)
        self.draw_text("2. Begin Processing", self.font, option2_color, option2_y)
        
        # Draw instructions
        instructions_y = self.screen_height // 2 + 100
        self.draw_text("Press 1 or 2 to select, or use arrow keys and spacebar", self.small_font, TEXT_COLOR, instructions_y)
        
        # Draw transcription status if active
        status = self.transcription_worker.get_status()
        if status['queue_size'] > 0 or status['current']:
            status_y = self.screen_height - 100
            
            if status['current']:
                self.draw_text(f"Transcribing: {status['current']}", self.small_font, (255, 255, 0), status_y)
            
            if status['queue_size'] > 0:
                queue_y = status_y + 30
                self.draw_text(f"Queue: {status['queue_size']} files pending", self.small_font, (255, 200, 0), queue_y)
                
            # Warning about exiting
            warning_y = status_y + 60
            self.draw_text("Please wait before exiting - transcription in progress", self.small_font, (255, 100, 100), warning_y)
        
    def draw_target_selection(self):
        """Draw the target selection screen"""
        # Draw title
        title_y = self.screen_height // 2 - 200
        self.draw_text("Select Target Image", self.menu_font, TEXT_COLOR, title_y)
        
        if not self.target_files:
            # No target files available
            no_files_y = self.screen_height // 2
            self.draw_text("No target images found. Create one first.", self.font, TEXT_COLOR, no_files_y)
            return_y = self.screen_height // 2 + 50
            self.draw_text("Press ESCAPE to return to menu", self.small_font, TEXT_COLOR, return_y)
        else:
            # Show available target files
            start_y = self.screen_height // 2 - 100
            for i, filename in enumerate(self.target_files):
                y_pos = start_y + (i * 40)
                color = MENU_HIGHLIGHT_COLOR if i == self.selected_target else MENU_OPTION_COLOR
                display_name = filename.replace('.txt', '').replace('_', ' ')
                self.draw_text(f"{i + 1}. {display_name}", self.font, color, y_pos)
            
            # Draw instructions
            instructions_y = self.screen_height // 2 + 100
            self.draw_text("Use arrow keys and spacebar to select", self.small_font, TEXT_COLOR, instructions_y)
        
    def draw_recording(self):
        """Draw the recording interface"""
        # Draw current question
        question_y = self.screen_height // 2 - 100
        if self.state == "recording":
            self.draw_text(TARGET_QUESTIONS[self.current_question], self.font, TEXT_COLOR, question_y)
        
        # Draw recording status
        recording_y = self.screen_height // 2
        self.draw_text(RECORDING_TEXT, self.font, (255, 0, 0), recording_y)  # Red text for recording
        
        # Draw pulsing circle to indicate recording
        pulse_time = time.time() * 2  # Faster pulse
        pulse_alpha = int((math.sin(pulse_time) + 1) * 127.5)  # 0-255
        circle_x = self.screen_width // 2
        circle_y = self.screen_height // 2 + 60
        
        circle_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (255, 0, 0, pulse_alpha), (20, 20), 20)
        self.screen.blit(circle_surface, (circle_x - 20, circle_y - 20))
        
    def draw_transcribing(self):
        """Draw the transcribing interface"""
        # Draw transcribing message
        transcribing_y = self.screen_height // 2
        self.draw_text(TRANSCRIBING_TEXT, self.font, TEXT_COLOR, transcribing_y)
        
        # Draw loading animation
        loading_y = self.screen_height // 2 + 50
        dots = "." * ((int(time.time() * 2) % 4))
        self.draw_text(dots, self.font, TEXT_COLOR, loading_y)
        
    def handle_menu_state(self):
        """Handle the main menu state"""
        self.draw_menu()
        
    def handle_target_identification_state(self):
        """Handle the target identification state"""
        # Check if we need to play the audio for this question
        if not self.audio_played:
            # Play the audio file for the current question
            if self.audio_handler.play_audio_file(QUESTION_AUDIO_FILES[self.current_question]):
                self.audio_played = True
            else:
                # If audio fails, go straight to recording
                self.state = "recording"
                self.audio_recorder.start_recording()
                
        # Check if audio has finished playing
        elif not self.audio_handler.is_audio_playing():
            # Audio finished, move to recording
            self.state = "recording"
            self.audio_recorder.start_recording()
            
        # Draw waiting message while audio is playing
        if self.audio_played and self.audio_handler.is_audio_playing():
            waiting_y = self.screen_height // 2
            self.draw_text("Please listen...", self.font, TEXT_COLOR, waiting_y)
            
    def handle_recording_state(self):
        """Handle the recording state"""
        self.draw_recording()
        
    def handle_transcribing_state(self):
        """Handle the transcribing state"""
        self.draw_transcribing()
        
        # This is handled by the event system when transcription completes
        
    def handle_target_selection_state(self):
        """Handle the target selection state"""
        self.draw_target_selection()

    # cue_in_states branch edits #1
    def save_cue_in_script(script, target_filename):
        """Save cue-in script to a file for editing before audio is generated"""
        # Create cue in script folder
        cue_scripts_dir = "cue_in_scripts"
        os.makedirs(cue_scripts_dir, exist_ok=True)

        # Extract target number from filename
        target_num = target_filename.split('_')[2].split('.')[0]
        script_filename = os.path.join(cue_scripts_dir, f"Cue_In_Script_{target_num}.txt")

        try:
            with open(script_filename, 'w') as f:
                f.write(f"Cue-In Script for {target_filename}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Script:\n")
                f.write("-" * 20 + "\n")
                f.write(script)
            
            print(f"Cue in script saved to: {script_filename}")
            return script_filename
        
        except Exception as e:
            print(f"Error saving cue in script: {e}")
            return None
        
    # cue_in_states branch edits #2
    def load_cue_in_script(script_filename):
        """Load cue in script from file"""
        try:
            with open(script_filename, 'r') as f:
                content = f.read()
            
            # Extract script part after metadata
            script_start = content.find("Script:\n" + "-" * 20 + "\n")     # find this, returns -1 if it cant
            if script_start != -1:
                script_start += len("Script:\n" + "-" * 20 + "\n")
                script = content[script_start:].strip()
                return script
        except Exception as e:
            print(f"Error loading cue in script: {e}")
            return ""
    
    # cue_in_states branch edits #3
    def handle_cue_in_generate_state(self):
        """Handle the cue in script generation state"""
        if not self.cue_in_generated:
            # Load selected target file
            if self.target_files and self.selected_target < len(self.target_files):
                filename = self.target_files[self.selected_target]
                responses = load_target_responses(filename)
            
                if responses:
                    # Generate cue-in script
                    self.cue_in_script = self.llm_handler.generate_cue_in_script(responses)
                
                    # Save script to file for editing
                    self.cue_in_script_file = save_cue_in_script(self.cue_in_script, filename)
                
                    if self.cue_in_script_file:
                        self.cue_in_generated = True
                        print(f"Cue-in script generated and saved to: {self.cue_in_script_file}")
                    
        # Draw generation status
        if self.cue_in_generated:
            # Script generated, show instructions
            title_y = self.screen_height // 2 - 100
            self.draw_text("Cue-In Script Generated", self.font, TEXT_COLOR, title_y)
        
            instruction1_y = self.screen_height // 2 - 20
            instruction2_y = self.screen_height // 2 + 20
            instruction3_y = self.screen_height // 2 + 60
        
            self.draw_text(f"Script saved to: {os.path.basename(self.cue_in_script_file)}", 
                        self.small_font, TEXT_COLOR, instruction1_y)
            self.draw_text("Edit the script file if needed, then:", 
                        self.small_font, TEXT_COLOR, instruction2_y)
            self.draw_text("Press SPACEBAR to continue with audio generation", 
                        self.small_font, (255, 255, 0), instruction3_y)
        else:
            # Still generating
            waiting_y = self.screen_height // 2
            self.draw_text("Generating cue-in script...", self.font, TEXT_COLOR, waiting_y)
        
            # Add loading dots
            loading_y = self.screen_height // 2 + 50
            dots = "." * ((int(time.time() * 2) % 4))
            self.draw_text(dots, self.font, TEXT_COLOR, loading_y)

    # cue_in_states branch edit #4
    def handle_cue_in_review_state(self):
        """Handle the cue-in script review state"""
        # Load the (possibly edited) script from file
        if hasattr(self, 'cue_in_script_file') and os.path.exists(self.cue_in_script_file):
            edited_script = load_cue_in_script(self.cue_in_script_file)
        
            # Display the script for review
            title_y = self.screen_height // 2 - 150
            self.draw_text("Review Cue-In Script", self.font, TEXT_COLOR, title_y)
        
            # Show first few lines of the script
            lines = edited_script.split('\n')[:4]  # Show first 4 lines
            start_y = self.screen_height // 2 - 80
            for i, line in enumerate(lines):
                if line.strip():
                # Truncate long lines for display
                    display_line = line.strip()
                    if len(display_line) > 60:
                        display_line = display_line[:57] + "..."
                    self.draw_text(display_line, self.small_font, TEXT_COLOR, start_y + i * 25)
        
            # Show instructions
            instruction1_y = self.screen_height // 2 + 50
            instruction2_y = self.screen_height // 2 + 80
            instruction3_y = self.screen_height // 2 + 110
        
            self.draw_text("Press SPACEBAR to generate audio and continue", 
                        self.small_font, (255, 255, 0), instruction1_y)
            self.draw_text("Press E to edit the script file again", 
                        self.small_font, TEXT_COLOR, instruction2_y)
            self.draw_text("Press ESC to return to menu", 
                        self.small_font, TEXT_COLOR, instruction3_y)
        else:
            error_y = self.screen_height // 2
            self.draw_text("Error: Could not load cue-in script file", self.font, (255, 0, 0), error_y)

    def handle_cue_in_audio_state(self):
        """Handle the cue-in audio generation and playback state"""
        if not hasattr(self, 'cue_in_audio_generated'):
            self.cue_in_audio_generated = False
    
        if not self.cue_in_audio_generated:
            # Load the final script and generate audio
            if hasattr(self, 'cue_in_script_file') and os.path.exists(self.cue_in_script_file):
                final_script = load_cue_in_script(self.cue_in_script_file)
            
                # Generate audio for the final script
                cue_in_path = os.path.join(AUDIO_DIR, CUE_IN_AUDIO)
                if self.audio_handler.generate_elevenlabs_audio(final_script, cue_in_path):
                    # Play the cue-in audio
                    self.audio_handler.play_audio_file(CUE_IN_AUDIO)
                    self.cue_in_audio_generated = True
                    self.cue_in_script = final_script  # Update the script variable
                    print("Cue-in audio generated and playing")
                else:
                    # If audio generation fails, show error
                    error_y = self.screen_height // 2
                    self.draw_text("Error generating audio. Press ESC to return.", 
                                self.font, (255, 0, 0), error_y)
                    return
            else:
                error_y = self.screen_height // 2
                self.draw_text("Error: Script file not found", self.font, (255, 0, 0), error_y)
                return
    
        # Check if audio has finished playing
        if self.cue_in_audio_generated and not self.audio_handler.is_audio_playing():
            # Audio complete, start processing
            self.state = "oscillating"
            self.current_cycle = 0
            self.start_time = time.time()
            self.audio_played_for_cycle = False
            # Reset audio generation flag for next session
            self.cue_in_audio_generated = False
            print("Starting processing after cue-in")
            return
        
        # Draw cue-in script or status
        if self.cue_in_audio_generated:
            if hasattr(self, 'cue_in_script') and self.cue_in_script:
                # Show first few lines of the script
                lines = self.cue_in_script.split('\n')[:3]  # Show first 3 lines
                start_y = self.screen_height // 2 - 50
                for i, line in enumerate(lines):
                    if line.strip():
                        self.draw_text(line.strip(), self.small_font, TEXT_COLOR, start_y + i * 30)
                    
                if self.audio_handler.is_audio_playing():
                    status_y = self.screen_height // 2 + 100
                    self.draw_text("Please listen and follow along...", self.small_font, TEXT_COLOR, status_y)
        else:
            waiting_y = self.screen_height // 2
            self.draw_text("Generating cue-in audio...", self.font, TEXT_COLOR, waiting_y)


    def handle_oscillation_state(self):
        """Handle the oscillating circle state"""
        # Check if oscillation time has elapsed
        elapsed_time = time.time() - self.start_time
    
        if elapsed_time >= OSCILLATION_DURATION:
            # Check if we need to play audio
            if not self.audio_played_for_cycle:
                # Play audio
                self.audio_played_for_cycle = True
                self.audio_handler.play_audio_file(WHAT_NOTICED_AUDIO)
                print(f"Cycle {self.current_cycle + 1}: Playing 'What did you notice?' audio")
        
            # Check if audio finished playing, then auto-start recording
            elif not self.audio_handler.is_audio_playing():
                self.state = "processing_recording"
                self.audio_recorder.start_recording()
                self.audio_played_for_cycle = False  # Reset for next cycle
                print(f"Auto-started recording for cycle {self.current_cycle + 1}")
        else:
            # Continue oscillating - draw the circle
            self.draw_circle()
            
    def handle_waiting_state(self):
        """skipping this step"""
        # 
        # Draw the prompt text
        prompt_y = self.screen_height // 2 - 50
        continue_y = self.screen_height // 2 + 20
        
        self.draw_text(PROMPT_TEXT, self.font, TEXT_COLOR, prompt_y)
        self.draw_text(CONTINUE_TEXT, self.small_font, TEXT_COLOR, continue_y)
        
    def handle_processing_recording_state(self):
        """Handle recording during processing phase"""
        # Draw prompt
        prompt_y = self.screen_height // 2 - 100
        self.draw_text(PROMPT_TEXT, self.font, TEXT_COLOR, prompt_y)
        
        # Draw recording status
        self.draw_recording()
        
    def handle_feedback_state(self):
        """Handle the feedback display state"""
        # Check if feedback time has elapsed
        elapsed_time = time.time() - self.feedback_start_time
        
        if elapsed_time >= FEEDBACK_DISPLAY_TIME:
            # Feedback time is up, start fading
            self.state = "fading"
            self.fade_start_time = time.time()
            print(f"Cycle {self.current_cycle + 1}: Feedback complete, starting fade")
        else:
            # Continue showing feedback text
            feedback_y = self.screen_height // 2 - 25
            self.draw_text(FEEDBACK_TEXT, self.font, TEXT_COLOR, feedback_y)
            
    def handle_fading_state(self):
        """Handle the fading transition state"""
        # Calculate fade progress (0 to 1)
        elapsed_time = time.time() - self.fade_start_time
        fade_progress = elapsed_time / FADE_DURATION
        
        if fade_progress >= 1.0:
            # Fade complete, start next cycle or end program
            self.current_cycle += 1
            
            if self.current_cycle >= TOTAL_CYCLES:
                # All cycles complete
                print("All cycles complete. Session will continue processing in background.")
                
                # Update session status
                metadata_path = os.path.join(self.current_session_path, "metadata.json")
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata["status"] = "completed"
                metadata["completed_at"] = time.time()
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Note: Responses are saved progressively by the worker thread
                
                # Reset for next session
                self.current_session_id = None
                self.current_session_path = None
                self.processing_responses = None
                self.state = "menu"
                self.current_cycle = 0
                return True
            else:
                # Start next cycle
                self.state = "oscillating"
                self.start_time = time.time()
                self.audio_played_for_cycle = False
                print(f"Starting cycle {self.current_cycle + 1}")
        else:
            # Continue fading - calculate alpha value
            alpha = int(255 * (1.0 - fade_progress))  # Fade from 255 to 0
            
            # Draw feedback text with fading alpha
            feedback_y = self.screen_height // 2 - 25
            
            # Create a surface for the text with alpha
            text_surface = self.font.render(FEEDBACK_TEXT, True, TEXT_COLOR)
            text_surface.set_alpha(alpha)
            
            # Center and draw the fading text
            text_rect = text_surface.get_rect()
            text_rect.centerx = self.screen_width // 2
            text_rect.y = feedback_y
            self.screen.blit(text_surface, text_rect)
            
        return True
        
    def process_transcription_complete(self, transcription):
        """Process completed transcription for target identification only"""
        if self.state == "transcribing":
            # Target identification phase
            self.target_responses.append(transcription)
            print(f"Response {len(self.target_responses)}: {transcription}")
            
            # Move to next question or finish
            self.current_question += 1
            if self.current_question >= len(TARGET_QUESTIONS):
                # All questions answered, save and return to menu
                filename = save_target_responses(self.target_responses)
                if filename:
                    print(f"Target identification complete. Saved to {filename}")
                self.state = "menu"
                self.current_question = 0
                self.target_responses = []
                self.audio_played = False
            else:
                # Move to next question
                self.audio_played = False
                self.state = "target_identification"
            
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            # transcription complete event
            if event.type == pygame.USEREVENT + 1:
                if self.state == "transcribing":
                    self.process_transcription_complete(event.transcription)
                continue
            # Check for quit events
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                # Handle escape key to quit or return to menu
                if event.key == pygame.K_ESCAPE:
                    if self.state == "menu":
                        return False
                    else:
                        # Stop any recording in progress
                        if self.state in ["recording", "processing_recording"]:
                            self.audio_recorder.stop_recording()
                        # Return to menu from any other state
                        self.state = "menu"
                        self.audio_handler.stop_audio()
                        
                # Handle menu navigation
                elif self.state == "menu":
                    if event.key == pygame.K_1:
                        # Start target identification
                        self.state = "target_identification"
                        self.current_question = 0
                        self.target_responses = []
                        self.audio_played = False
                        print("Starting target identification")
                    elif event.key == pygame.K_2:
                        # Start target selection for processing
                        self.target_files = get_existing_target_files()
                        if self.target_files:
                            self.state = "target_selection"
                            self.selected_target = 0
                            print("Starting target selection")
                        else:
                            print("No target files found. Create one first.")
                    elif event.key == pygame.K_UP:
                        self.menu_selected = 0
                    elif event.key == pygame.K_DOWN:
                        self.menu_selected = 1
                    elif event.key == pygame.K_SPACE:
                        if self.menu_selected == 0:
                            self.state = "target_identification"
                            self.current_question = 0
                            self.target_responses = []
                            self.audio_played = False
                            print("Starting target identification")
                        else:
                            self.target_files = get_existing_target_files()
                            if self.target_files:
                                self.state = "target_selection"
                                self.selected_target = 0
                                print("Starting target selection")
                            else:
                                print("No target files found. Create one first.")
                                
                # Handle target selection navigation
                elif self.state == "target_selection":
                    if event.key == pygame.K_UP and self.selected_target > 0:
                        self.selected_target -= 1
                    elif event.key == pygame.K_DOWN and self.selected_target < len(self.target_files) - 1:
                        self.selected_target += 1
                    elif event.key == pygame.K_SPACE:
                        if self.target_files:
                            # Create new processing session
                            self.current_session_id, self.current_session_path = self.session_manager.create_session()
                            self.processing_responses = ProcessingResponseList()
                            print(f"Created new session: {self.current_session_id}")
                            
                            # Start cue-in process
                            self.state = "cue_in_generate"      # cue_in -> cue_in_generate
                            self.cue_in_generated = False
                            self.cue_in_script = ""
                            print(f"Selected target: {self.target_files[self.selected_target]}")
                
                # Handle cue-in generation
                elif self.state == "cue_in_generate":
                    if event.key == pygame.K_SPACE and self.cue_in_generated:
                        # Move to review state
                        self.state = "cue_in_review"
                        print("Moving to cue in script review")

                # Handle cue-in review  
                elif self.state == "cue_in_review":
                    if event.key == pygame.K_SPACE:
                        # Move to audio generation
                        self.state = "cue_in_audio"
                        print("Generating cue-in audio")
                    elif event.key == pygame.K_e:
                        # Open script file for editing (this will depend on your OS)
                        if hasattr(self, 'cue_in_script_file'):
                            print(f"Please edit the file: {self.cue_in_script_file}")
                            print("Then press SPACEBAR to continue")
                            # Optionally, try to open the file with the default editor:
                            try:
                                import subprocess
                                import platform
                                if platform.system() == "Windows":
                                    subprocess.run(["notepad", self.cue_in_script_file])
                                elif platform.system() == "Darwin":  # macOS
                                    subprocess.run(["open", "-t", self.cue_in_script_file])
                                else:  # Linux
                                    subprocess.run(["xdg-open", self.cue_in_script_file])
                            except Exception as e:
                                print(f"Could not auto-open editor. Please manually edit: {self.cue_in_script_file}")

                            
                # Handle recording stop
                elif self.state == "recording" and event.key == pygame.K_SPACE:
                    # Stop recording and start transcription
                    audio_file = self.audio_recorder.stop_recording()
                    print(f"debug: audio file returned: {audio_file}")
                    if audio_file:
                        self.state = "transcribing"
                        # Start transcription in a separate thread
                        threading.Thread(
                            target=self._transcribe_audio,
                            args=(audio_file,)
                        ).start()
                        
                # Handle processing recording start
                elif self.state == "waiting" and event.key == pygame.K_SPACE:
                    # Start recording for processing response
                    self.state = "processing_recording"
                    self.audio_recorder.start_recording()
                    print(f"Recording response for cycle {self.current_cycle + 1}")
                    
                # Handle processing recording stop
                elif self.state == "processing_recording" and event.key == pygame.K_SPACE:
                    # Stop recording and immediately continue to feedback
                    audio_file = self.audio_recorder.stop_recording()
                    if audio_file:
                        # Add to background queue for transcription
                        self.transcription_worker.add_audio(
                            audio_file, 
                            self.current_cycle + 1,
                            self.current_session_path,
                            self.processing_responses
                        )
                        # Immediately go to feedback - no waiting for transcription
                        self.state = "feedback"
                        self.feedback_start_time = time.time()
                        print(f"Recording complete for cycle {self.current_cycle + 1}, continuing immediately")
                    
        return True
        
    def _transcribe_audio(self, audio_file):
        """Transcribe audio in a separate thread"""
        try:
            transcription = self.audio_recorder.transcribe_audio(audio_file)

        
            # Clean up audio file
            try:
                os.remove(audio_file)
            except:
                pass
            
            # Process transcription on main thread
            #self.process_transcription_complete(transcription)
            pygame.event.post(pygame.event.Event(
                pygame.USEREVENT + 1,
                {'transcription': transcription}
            ))
        
        except Exception as e:
            print(f"Error in transcription thread: {e}")
            # post empty transcription on error
            pygame.event.post(pygame.event.Event(
                pygame.USEREVENT + 1,
                {'transcription': transcription}
            ))

    def setup_audio_files(self):
        """Setup audio files on first run"""
        print("Setting up audio files...")
        if not self.audio_handler.generate_question_audio_files():
            print("Warning: Could not generate all audio files. Some features may not work.")
            
    def run(self):
        """Main program loop"""
        print("Starting EMDR Therapy Program with Voice Recording")
        print("Press ESCAPE to quit at any time")
        
        # Generate audio files if needed
        self.setup_audio_files()
        
        print("Use menu to select Target Identification or Begin Processing")
        
        # Main game loop
        running = True
        while running:
            # Handle user input
            running = self.handle_events()
            if not running:
                break
                
            # Clear screen with background color
            self.screen.fill(BACKGROUND_COLOR)
            
            # Handle current state
            if self.state == "menu":
                self.handle_menu_state()
            elif self.state == "target_identification":
                self.handle_target_identification_state()
            elif self.state == "recording":
                self.handle_recording_state()
            elif self.state == "transcribing":
                self.handle_transcribing_state()
            elif self.state == "target_selection":
                self.handle_target_selection_state()
            elif self.state == "cue_in_generate":
                self.handle_cue_in_generate_state()
            elif self.state == "cue_in_review":
                self.handle_cue_in_review_state()
            elif self.state == "cue_in_audio":
                self.handle_cue_in_audio_state()
            elif self.state == "cue_in":
                self.handle_cue_in_state()
            elif self.state == "oscillating":
                self.handle_oscillation_state()
            elif self.state == "waiting":   #remove
                self.handle_waiting_state()
            elif self.state == "processing_recording":
                self.handle_processing_recording_state()
            elif self.state == "feedback":
                self.handle_feedback_state()
            elif self.state == "fading":
                running = self.handle_fading_state()
                
            # Update display
            pygame.display.flip()
            
            # Control frame rate (60 FPS for smooth animation)
            self.clock.tick(60)
            
        # Clean up
        self.transcription_worker.stop()
        self.audio_handler.stop_audio()
        self.audio_recorder.cleanup()
        pygame.quit()
        sys.exit()

# ===== PROGRAM ENTRY POINT =====
if __name__ == "__main__":
    # Create and run the EMDR program
    program = EMDRProgram()
    program.run()