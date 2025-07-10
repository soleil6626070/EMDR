import pygame
import sys
import time
import math

# ===== CONFIGURATION SECTION - All adjustable parameters =====
# Visual parameters
CIRCLE_COLOR = (255, 0, 0)  # Red color in RGB format
CIRCLE_DIAMETER_CM = 2.0    # Circle diameter in centimeters
BACKGROUND_COLOR = (80, 80, 80)  # Light grey background in RGB
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
TEXT_COLOR = (0, 0, 0)  # Black text
FONT_SIZE = 36
SMALL_FONT_SIZE = 24

# Timing parameters
FEEDBACK_DISPLAY_TIME = 2.0  # How long "Notice that" is shown in seconds
FADE_DURATION = 1.0          # How long the fade effect takes in seconds

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
        
        # Initialize program state
        self.current_cycle = 0
        self.state = "oscillating"  # Can be: "oscillating", "waiting", "feedback", "fading"
        self.start_time = time.time()
        self.feedback_start_time = 0
        self.fade_start_time = 0
        
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
                # All cycles complete, end program
                print("All cycles complete. Ending program.")
                return False
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
                # Handle spacebar in waiting state
                elif event.key == pygame.K_SPACE and self.state == "waiting":
                    # User pressed spacebar, transition to feedback
                    self.state = "feedback"
                    self.feedback_start_time = time.time()
                    print(f"Cycle {self.current_cycle + 1}: User input received, showing feedback")
                    
        return True
        
    def run(self):
        """Main program loop"""
        print("Starting EMDR Therapy Program")
        print(f"Total cycles: {TOTAL_CYCLES}")
        print(f"Oscillation duration: {OSCILLATION_DURATION} seconds")
        print(f"Press ESCAPE to quit at any time")
        print(f"Starting cycle 1")
        
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
            if self.state == "oscillating":
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
        pygame.quit()
        sys.exit()

# ===== PROGRAM ENTRY POINT =====
if __name__ == "__main__":
    # Create and run the EMDR program
    program = EMDRProgram()
    program.run()