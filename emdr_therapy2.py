import pygame
import sys
import time
import math
import pyttsx3
import threading
import os


# ===== CONFIGURATION SECTION - All adjustable parameters =====
# Visual parameters
CIRCLE_COLOR = (255, 0, 0)  # Red color in RGB format
CIRCLE_DIAMETER_CM = 2.0    # Circle diameter in centimeters
BACKGROUND_COLOR = (80, 80, 80)  # Dark grey background in RGB
MARGIN_CM = 1.0             # Margin from screen edges in centimeters

# Animation parameters
OSCILLATIONS_PER_SECOND = 1.2  # How many complete left-right-left cycles per second
OSCILLATION_DURATION = 30.0    # How long the circle oscillates in seconds

# Session parameters
TOTAL_CYCLES = 10  # Total number of oscillation-feedback cycles

# Text parameters
PROMPT_TEXT = "What did you notice?"
CONTINUE_TEXT = "(tap spacebar to continue)"
FEEDBACK_TEXT = "Notice that"
TEXT_COLOR = (255, 255, 255)  # White text for dark background
FONT_SIZE = 36
SMALL_FONT_SIZE = 24
MENU_FONT_SIZE = 48

# Text input parameters
INPUT_BOX_COLOR = (255, 255, 255)  # White background for text input
INPUT_TEXT_COLOR = (0, 0, 0)  # Black text in input box
INPUT_BOX_WIDTH = 800
INPUT_BOX_HEIGHT = 50
MAX_INPUT_LENGTH = 1000

# Menu parameters
MENU_OPTION_COLOR = (200, 200, 200)  # Light grey for menu options
MENU_HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow for highlighted option

# Timing parameters
FEEDBACK_DISPLAY_TIME = 2.0  # How long "Notice that" is shown in seconds
FADE_DURATION = 1.0          # How long the fade effect takes in seconds

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

# ===== TTS HANDLER CLASS =====
class TTSHandler:
    def __init__(self):
        """Initialize the text-to-speech handler"""
        # Initialize the TTS engine
        self.engine = pyttsx3.init()
        
        # Set up TTS properties (can be adjusted later)
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level
        
        # Track if TTS is currently speaking
        self.is_speaking = False
        self.speech_thread = None
        
    def speak(self, text):
        """Speak the given text and wait for completion"""
        # Don't start new speech if already speaking
        if self.is_speaking:
            return
            
        # Set speaking flag
        self.is_speaking = True
        
        # Create and start speech thread
        self.speech_thread = threading.Thread(target=self._speak_thread, args=(text,))
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
    def _speak_thread(self, text):
        """Internal method to handle speech in separate thread"""
        try:
            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            # Reset speaking flag
            self.is_speaking = False
            
    def is_finished(self):
        """Check if TTS has finished speaking"""
        return not self.is_speaking
        
    def stop(self):
        """Stop the TTS engine"""
        if self.speech_thread and self.speech_thread.is_alive():
            self.engine.stop()
        self.is_speaking = False

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
        
        # Initialize TTS handler
        self.tts = TTSHandler()
        
        # Initialize program state - start with menu
        self.current_cycle = 0
        self.state = "menu"  # States: menu, target_identification, text_input, oscillating, waiting, feedback, fading
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
        self.question_spoken = False
        
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
        # Check if we need to speak the initial prompt
        if self.current_question == 0 and not self.question_spoken:
            intro_text = "We are going to identify a target image to act as a starting point for the processing. When you're ready, I'd like you to visualise the most significant part of the trauma, like a snapshot or freeze frame of the most intense moment of that traumatic memory."
            self.tts.speak(intro_text)
            self.question_spoken = True
            
        # Check if we need to move to text input
        elif self.tts.is_finished() and not self.input_active:
            self.state = "text_input"
            self.input_active = True
            self.current_input = ""
            
        # Draw waiting message while TTS is speaking
        if not self.tts.is_finished():
            waiting_y = self.screen_height // 2
            self.draw_text("Please listen...", self.font, TEXT_COLOR, waiting_y)
            
    def handle_text_input_state(self):
        """Handle the text input state"""
        self.draw_text_input()
        
    def handle_oscillation_state(self):
        """Handle the oscillating circle state"""
        # Check if oscillation time has elapsed
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= OSCILLATION_DURATION:
            # Time's up, transition to waiting state
            self.state = "waiting"
            print(f"Cycle {self.current_cycle + 1}: Oscillation complete, waiting for user input")
        else:
            # Continue oscillating - draw the circle
            self.draw_circle()
            
    def handle_waiting_state(self):
        """Handle the waiting for user input state"""
        # Draw the prompt text
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
                # Handle escape key to quit
                if event.key == pygame.K_ESCAPE:
                    return False
                    
                # Handle menu navigation
                elif self.state == "menu":
                    if event.key == pygame.K_1:
                        # Start target identification
                        self.state = "target_identification"
                        self.current_question = 0
                        self.target_responses = []
                        self.question_spoken = False
                        print("Starting target identification")
                    elif event.key == pygame.K_2:
                        # Start processing
                        self.state = "oscillating"
                        self.current_cycle = 0
                        self.start_time = time.time()
                        print("Starting processing")
                    elif event.key == pygame.K_UP:
                        self.menu_selected = 0
                    elif event.key == pygame.K_DOWN:
                        self.menu_selected = 1
                    elif event.key == pygame.K_SPACE:
                        if self.menu_selected == 0:
                            self.state = "target_identification"
                            self.current_question = 0
                            self.target_responses = []
                            self.question_spoken = False
                            print("Starting target identification")
                        else:
                            self.state = "oscillating"
                            self.current_cycle = 0
                            self.start_time = time.time()
                            print("Starting processing")
                            
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
                        else:
                            # Move to next question
                            self.current_input = ""
                            self.tts.speak(TARGET_QUESTIONS[self.current_question])
                            
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
        
    def run(self):
        """Main program loop"""
        print("Starting EMDR Therapy Program")
        print("Press ESCAPE to quit at any time")
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
        self.tts.stop()
        pygame.quit()
        sys.exit()

# ===== PROGRAM ENTRY POINT =====
if __name__ == "__main__":
    # Create and run the EMDR program
    program = EMDRProgram()
    program.run()
