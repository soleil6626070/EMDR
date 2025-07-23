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

import openai
# import anthropic

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
OSCILLATION_DURATION = 30.0    # Cycle duration - Ask therapist

# Session parameters
TOTAL_CYCLES = 10

# Text parameters
PROMPT_TEXT = "What did you notice?"
CONTINUE_TEXT = "(tap spacebar to continue)"
FEEDBACK_TEXT = "Notice that"
TEXT_COLOR = (255, 255, 255)  # White 
FONT_SIZE = 36
SMALL_FONT_SIZE = 24
MENU_FONT_SIZE = 48

FEEDBACK_DISPLAY_TIME = 2.0  # How long "Notice that" is shown in seconds
FADE_DURATION = 1.0          # Length of fade effect

# Text input parameters
INPUT_BOX_COLOR = (255, 255, 255)  # White 
INPUT_TEXT_COLOR = (0, 0, 0)       # Black 
INPUT_BOX_WIDTH = 800
INPUT_BOX_HEIGHT = 50
MAX_INPUT_LENGTH = 1000

# Menu parameters
MENU_OPTION_COLOR = (200, 200, 200)  # Light grey 
MENU_HIGHLIGHT_COLOR = (255, 255, 0)  # Highlighted option - yellow


# Audio file paths
AUDIO_DIR = "audio_files"
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
    """Get the next available target image filename"""
    # Start with Target_Image_1.txt and increment until we find an unused name
    counter = 1
    while True:
        filename = f"Target_Image_{counter}.txt"
        if not os.path.exists(filename):
            return filename
        counter += 1

def get_existing_target_files():
    """Get a list of existing target image files"""
    files = []
    for file in os.listdir('.'):
        if file.startswith('Target_Image_') and file.endswith('.txt'):
            files.append(file)
    return sorted(files)

def save_target_responses(responses):
    """Save target image responses to a file"""
    filename = get_next_target_filename()
    
    try:
        with open(filename, 'w') as f:
            f.write(f"Target Image {filename.split('_')[2].split('.')[0]}\n")
            f.write("=" * 50 + "\n\n")
            
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
        with open(filename, 'r') as f:
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
        self.llm_handler = LLMHandler()
        
        # Initialize program state - start with menu
        self.current_cycle = 0
        self.state = "menu"  # States: menu, target_identification, text_input, target_selection, cue_in, oscillating, waiting, feedback, fading
        self.start_time = time.time()
        self.feedback_start_time = 0
        self.fade_start_time = 0
        
        # Menu state variables
        self.menu_selected = 0  # 0 = Target ID, 1 = Begin Processing
        
        # Target identification state variables
        self.current_question = 0
        self.target_responses = []
        self.current_input = ""
        self.input_active = False
        self.audio_played = False
        
        # Target selection variables
        self.target_files = []
        self.selected_target = 0
        
        # Cue-in state variables
        self.cue_in_script = ""
        self.cue_in_generated = False
        
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
        
    def draw_text_input(self):
        """Draw the text input box and current question"""
        # Draw current question
        question_y = self.screen_height // 2 - 100
        self.draw_text(TARGET_QUESTIONS[self.current_question], self.font, TEXT_COLOR, question_y)
        
        # Calculate input box position
        input_x = (self.screen_width - INPUT_BOX_WIDTH) // 2
        input_y = self.screen_height // 2 - 25
        
        # Draw input box background
        input_rect = pygame.Rect(input_x, input_y, INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT)
        pygame.draw.rect(self.screen, INPUT_BOX_COLOR, input_rect)
        pygame.draw.rect(self.screen, TEXT_COLOR, input_rect, 2)  # Border
        
        # Draw current input text
        if self.current_input:
            # Handle text that might be too long for the box
            display_text = self.current_input
            text_surface = self.font.render(display_text, True, INPUT_TEXT_COLOR)
            
            # If text is too wide, show the end of the text
            if text_surface.get_width() > INPUT_BOX_WIDTH - 20:
                # Truncate from the beginning to show the end
                while text_surface.get_width() > INPUT_BOX_WIDTH - 20 and len(display_text) > 0:
                    display_text = display_text[1:]
                    text_surface = self.font.render(display_text, True, INPUT_TEXT_COLOR)
            
            # Draw the text
            text_y = input_y + (INPUT_BOX_HEIGHT - text_surface.get_height()) // 2
            self.screen.blit(text_surface, (input_x + 10, text_y))
        
        # Draw cursor if input is active
        if self.input_active:
            cursor_x = input_x + 10
            if self.current_input:
                text_surface = self.font.render(self.current_input, True, INPUT_TEXT_COLOR)
                cursor_x += text_surface.get_width()
            cursor_y = input_y + 10
            pygame.draw.line(self.screen, INPUT_TEXT_COLOR, (cursor_x, cursor_y), (cursor_x, cursor_y + INPUT_BOX_HEIGHT - 20), 2)
        
        # Draw instructions
        instructions_y = self.screen_height // 2 + 80
        self.draw_text("Type your response and press Enter to continue", self.small_font, TEXT_COLOR, instructions_y)
        
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
                # If audio fails, show text and allow continuation
                self.state = "text_input"
                self.input_active = True
                self.current_input = ""
                
        # Check if audio has finished playing
        elif not self.audio_handler.is_audio_playing():
            # Audio finished, move to text input
            self.state = "text_input"
            self.input_active = True
            self.current_input = ""
            
        # Draw waiting message while audio is playing
        if self.audio_played and self.audio_handler.is_audio_playing():
            waiting_y = self.screen_height // 2
            self.draw_text("Please listen...", self.font, TEXT_COLOR, waiting_y)
            
    def handle_text_input_state(self):
        """Handle the text input state"""
        self.draw_text_input()
        
    def handle_target_selection_state(self):
        """Handle the target selection state"""
        self.draw_target_selection()
        
    def handle_cue_in_state(self):
        """Handle the cue-in state"""
        # Generate cue-in script if not already done
        if not self.cue_in_generated:
            # Load selected target file
            if self.target_files and self.selected_target < len(self.target_files):
                filename = self.target_files[self.selected_target]
                responses = load_target_responses(filename)
                
                if responses:
                    # Generate cue-in script
                    self.cue_in_script = self.llm_handler.generate_cue_in_script(responses)
                    
                    # Generate audio for cue-in script
                    cue_in_path = os.path.join(AUDIO_DIR, CUE_IN_AUDIO)
                    if self.audio_handler.generate_elevenlabs_audio(self.cue_in_script, cue_in_path):
                        # Play the cue-in audio
                        self.audio_handler.play_audio_file(CUE_IN_AUDIO)
                        self.cue_in_generated = True
                    else:
                        # If audio generation fails, show text
                        self.cue_in_generated = True
                        
        # Check if audio has finished playing
        elif not self.audio_handler.is_audio_playing():
            # Cue-in complete, start processing
            self.state = "oscillating"
            self.current_cycle = 0
            self.start_time = time.time()
            print("Starting processing after cue-in")
            
        # Draw cue-in script or waiting message
        if self.cue_in_script:
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
            self.draw_text("Generating cue-in script...", self.font, TEXT_COLOR, waiting_y)
            
    def handle_oscillation_state(self):
        """Handle the oscillating circle state"""
        # Check if oscillation time has elapsed
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= OSCILLATION_DURATION:
            # Time's up, transition to waiting state
            self.state = "waiting"
            # Play "What did you notice?" audio
            self.audio_handler.play_audio_file(WHAT_NOTICED_AUDIO)
            print(f"Cycle {self.current_cycle + 1}: Oscillation complete, waiting for user input")
        else:
            # Continue oscillating - draw the circle
            self.draw_circle()
            
    def handle_waiting_state(self):
        """Handle the waiting for user input state"""
        # Draw the prompt text (no TTS, just text)
        prompt_y = self.screen_height // 2 - 50
        continue_y = self.screen_height // 2 + 20
        
        self.draw_text(PROMPT_TEXT, self.font, TEXT_COLOR, prompt_y)
        self.draw_text(CONTINUE_TEXT, self.small_font, TEXT_COLOR, continue_y)
        
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
                # All cycles complete, return to menu
                print("All cycles complete. Returning to menu.")
                self.state = "menu"
                self.current_cycle = 0
                return True
            else:
                # Start next cycle
                self.state = "oscillating"
                self.start_time = time.time()
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
        
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            # Check for quit events
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                # Handle escape key to quit or return to menu
                if event.key == pygame.K_ESCAPE:
                    if self.state == "menu":
                        return False
                    else:
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
                            # Start cue-in process
                            self.state = "cue_in"
                            self.cue_in_generated = False
                            self.cue_in_script = ""
                            print(f"Selected target: {self.target_files[self.selected_target]}")
                            
                # Handle text input
                elif self.state == "text_input":
                    if event.key == pygame.K_RETURN:
                        # Submit current response
                        self.target_responses.append(self.current_input)
                        print(f"Response {len(self.target_responses)}: {self.current_input}")
                        
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
                            self.current_input = ""
                            self.input_active = False
                            self.audio_played = False
                            self.state = "target_identification"
                            
                    elif event.key == pygame.K_BACKSPACE:
                        # Remove last character
                        self.current_input = self.current_input[:-1]
                    else:
                        # Add character to input (if printable and under limit)
                        if event.unicode.isprintable() and len(self.current_input) < MAX_INPUT_LENGTH:
                            self.current_input += event.unicode
                            
                # Handle spacebar in waiting state (during processing)
                elif event.key == pygame.K_SPACE and self.state == "waiting":
                    # User pressed spacebar, transition to feedback
                    self.state = "feedback"
                    self.feedback_start_time = time.time()
                    print(f"Cycle {self.current_cycle + 1}: User input received, showing feedback")
                    
        return True
        
    def setup_audio_files(self):
        """Setup audio files on first run"""
        print("Setting up audio files...")
        if not self.audio_handler.generate_question_audio_files():
            print("Warning: Could not generate all audio files. Some features may not work.")
            
    def run(self):
        """Main program loop"""
        print("Starting EMDR Therapy Program")
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
            elif self.state == "text_input":
                self.handle_text_input_state()
            elif self.state == "target_selection":
                self.handle_target_selection_state()
            elif self.state == "cue_in":
                self.handle_cue_in_state()
            elif self.state == "oscillating":
                self.handle_oscillation_state()
            elif self.state == "waiting":
                self.handle_waiting_state()
            elif self.state == "feedback":
                self.handle_feedback_state()
            elif self.state == "fading":
                running = self.handle_fading_state()
                
            # Update display
            pygame.display.flip()
            
            # Control frame rate (60 FPS for smooth animation)
            self.clock.tick(60)
            
        # Clean up
        self.audio_handler.stop_audio()
        pygame.quit()
        sys.exit()

# ===== PROGRAM ENTRY POINT =====
if __name__ == "__main__":
    # Create and run the EMDR program
    program = EMDRProgram()
    program.run()