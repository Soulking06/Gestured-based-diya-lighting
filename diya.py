import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Single hand is enough for this
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Colors
ORANGE = (255, 165, 0)
RED = (255, 69, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
DARK_GOLD = (184, 134, 11)
WOOD_COLOR = (210, 180, 140)
MATCH_HEAD_COLOR = (200, 50, 50)
MATCH_HEAD_BURNT = (50, 20, 20)
BLACK = (0, 0, 0)

# --- Hand Patterns ---
class HandPattern:
    UNKNOWN = 0
    MATCH_HOLD = 1  # Index extended only
    IGNITE = 2      # Thumb + Middle rub/snap
    SHIELD = 3      # Open palm (all fingers extended, fingers close)
    EXTINGUISH = 4  # Pinch (Thumb + Index touching)

class PatternRecognizer:
    def detect(self, landmarks):
        """
        Detects hand patterns based on landmarks.
        Returns: HandPattern ID
        """
        # Finger states (Extended or curled)
        # Tips: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
        # PIPs (Knuckles): Index=6, Middle=10, Ring=14, Pinky=18
        
        # Helper to check if finger is extended
        def is_extended(tip_idx, pip_idx):
            return landmarks[tip_idx].y < landmarks[pip_idx].y # Assuming upright hand
            
        # Refined extension check based on distance from wrist (0)
        # More robust against rotation than simple Y check
        wrist = landmarks[0]
        def is_finger_open(tip_idx, mcp_idx):
            return math.hypot(landmarks[tip_idx].x - wrist.x, landmarks[tip_idx].y - wrist.y) > \
                   math.hypot(landmarks[mcp_idx].x - wrist.x, landmarks[mcp_idx].y - wrist.y)

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        index_open = is_finger_open(8, 5)
        middle_open = is_finger_open(12, 9)
        ring_open = is_finger_open(16, 13)
        pinky_open = is_finger_open(20, 17)
        
        # 1. EXTINGUISH: Pinch (Thumb + Index touching)
        # Others can be open or closed, but focus is on the pinch
        pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
        if pinch_dist < 0.05: # Threshold for pinch
            return HandPattern.EXTINGUISH
            
        # 2. IGNITE: Thumb + Middle touching/rubbing
        # Index should be open (holding the stick)
        rub_dist = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)
        if index_open and rub_dist < 0.05:
            return HandPattern.IGNITE
            
        # 3. SHIELD: All fingers open and close together
        # (Simplified: Just check if all 4 fingers are open)
        if index_open and middle_open and ring_open and pinky_open:
            return HandPattern.SHIELD
            
        # 4. MATCH HOLD: Index open, others curl (or just Index open dominant)
        if index_open and not (middle_open and ring_open and pinky_open):
            return HandPattern.MATCH_HOLD
            
        return HandPattern.UNKNOWN

# --- Visualization Components ---

class StopButton:
    def __init__(self, x, y, size=60):
        self.rect = pygame.Rect(x, y, size, size)
        self.hovered = False

    def draw(self, surface):
        center = self.rect.center
        radius = self.rect.w // 2
        
        # Red Hexagon/Octagon for Stop Sign look
        points = []
        num_sides = 8
        for i in range(num_sides):
            angle = math.pi/8 + (2*math.pi * i) / num_sides
            px = center[0] + radius * math.cos(angle)
            py = center[1] + radius * math.sin(angle)
            points.append((px, py))
            
        color = (255, 50, 50) if self.hovered else (200, 0, 0)
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, WHITE, points, 3)
        
        # Inner "STOP" block (Square)
        rw, rh = 20, 20
        pygame.draw.rect(surface, WHITE, (center[0] - rw//2, center[1] - rh//2, rw, rh))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered: return True
        return False

class SmoothingFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = (
                self.value[0] * (1 - self.alpha) + new_value[0] * self.alpha,
                self.value[1] * (1 - self.alpha) + new_value[1] * self.alpha
            )
        return self.value

class Particle:
    def __init__(self, x, y, is_diya_flame=False):
        self.x = x
        self.y = y
        self.size = random.randint(12, 25) if is_diya_flame else random.randint(3, 8)
        self.life = 255
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-4, -1.5) if is_diya_flame else random.uniform(-3, -1)
        self.color_start = (255, 255, 150) 
        self.color_mid = (255, 140, 0)     
        self.color_end = (100, 20, 20)      
        self.is_diya_flame = is_diya_flame

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 6 if self.is_diya_flame else 15
        self.size -= 0.2
        if self.size < 0: self.size = 0

    def draw(self, surface):
        if self.life > 0 and self.size > 0:
            if self.life > 150: color = self.color_start
            elif self.life > 80: color = self.color_mid
            else: color = self.color_end
            
            particle_surf = pygame.Surface((int(self.size*2), int(self.size*2)), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, (*color, min(255, self.life)), (int(self.size), int(self.size)), int(self.size))
            surface.blit(particle_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_PREMULTIPLIED)

class FireSystem:
    def __init__(self):
        self.particles = []
    def emit(self, x, y, count=1, is_diya_flame=False):
        for _ in range(count): self.particles.append(Particle(x, y, is_diya_flame))
    def update(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles: p.update()
    def draw(self, surface):
        for p in self.particles: p.draw(surface)

class Matchstick:
    def __init__(self):
        self.active = False
        self.pos_filter = SmoothingFilter(0.3)
        self.pos = (0,0) # Center position
        self.head_pos = (0,0)
        self.angle = 0
        self.fire_system = FireSystem()
        self.is_lit = False
        self.burnt_amount = 0
        
        # Friction/Rub Logic
        self.rub_energy = 0
        self.prev_finger_pos = None
        self.pattern_recognizer = PatternRecognizer()

    def reset(self):
        self.is_lit = False
        self.burnt_amount = 0
        self.fire_system.particles = []
        self.rub_energy = 0

    def update(self, landmarks, w, h):
        # Recognize Pattern
        pattern = self.pattern_recognizer.detect(landmarks)
        
        # Landmarks needed
        thumb_tip = landmarks[4]
        index_base = landmarks[5]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Convert to pixels for Matchstick Physics
        # Note: Matchstick anchors to Index Finger
        ibx, iby = int(index_base.x * w), int(index_base.y * h)
        itx, ity = int(index_tip.x * w), int(index_tip.y * h)
        
        # VISUAL STABILIZATION:
        target_pos = (itx, ity)
        self.pos = self.pos_filter.update(target_pos)
        self.active = True
        
        # Calculate angle based on Index Finger
        dy = iby - ity
        dx = ibx - itx
        self.angle = -math.atan2(dy, dx) + math.pi/2 + math.pi 
        
        # Calculate Head Position
        length = 100
        p2_x = self.pos[0] + (length/2) * math.cos(self.angle)
        p2_y = self.pos[1] + (length/2) * math.sin(self.angle)
        self.head_pos = (p2_x, p2_y)

        # TRIGGER LOGIC with Patterns
        if not self.is_lit:
            if pattern == HandPattern.IGNITE:
                self.is_lit = True
        
        if self.is_lit:
            self.burnt_amount = min(1.0, self.burnt_amount + 0.005)

    def draw(self, surface):
        if not self.active: return

        cx, cy = self.pos
        length = 100
        thickness = 10
        
        p1_x = cx - (length/2) * math.cos(self.angle)
        p1_y = cy - (length/2) * math.sin(self.angle)
        p2_x = cx + (length/2) * math.cos(self.angle)
        p2_y = cy + (length/2) * math.sin(self.angle)
        
        # Shadow
        pygame.draw.line(surface, (0,0,0,80), (p1_x+5, p1_y+25), (p2_x+5, p2_y+25), thickness)
        # Stick
        pygame.draw.line(surface, WOOD_COLOR, (p1_x, p1_y), (p2_x, p2_y), thickness)
        
        # Head
        head_x = p2_x
        head_y = p2_y
        r = int(MATCH_HEAD_COLOR[0] * (1-self.burnt_amount) + MATCH_HEAD_BURNT[0] * self.burnt_amount)
        g = int(MATCH_HEAD_COLOR[1] * (1-self.burnt_amount) + MATCH_HEAD_BURNT[1] * self.burnt_amount)
        b = int(MATCH_HEAD_COLOR[2] * (1-self.burnt_amount) + MATCH_HEAD_BURNT[2] * self.burnt_amount)
        pygame.draw.circle(surface, (r, g, b), (int(head_x), int(head_y)), 9)
        
        if self.is_lit:
            self.fire_system.emit(head_x, head_y - 5, count=4)
            self.fire_system.update()
            self.fire_system.draw(surface)
            
            # Draw sparks/smoke if just ignited? Optional.
        
        self.active = False
        # visual feedback for rub energy (optional debugging)
        # if self.rub_energy > 0 and not self.is_lit:
        #    pygame.draw.circle(surface, (255, 255, 255), (int(cx), int(cy)), int(self.rub_energy/5), 1)

class Diya:
    def __init__(self, screen_w, screen_h):
        self.center = (screen_w // 2, screen_h // 2 + 50)
        self.is_lit = False
        self.wick_pos = (self.center[0] + 95, self.center[1] - 25) 
        self.flame_system = FireSystem()
        self.glow_radius = 0

    def reset(self):
        self.is_lit = False
        self.flame_system.particles = []

    def check_ignition(self, match_pos, is_match_lit):
        if self.is_lit: return
        if not is_match_lit: return
        
        # Check distance to wick
        dist = math.hypot(match_pos[0] - self.wick_pos[0], match_pos[1] - self.wick_pos[1])
        if dist < 60:
            self.is_lit = True

    def handle_interaction(self, pattern, hand_pos):
        if not self.is_lit: return

        # EXTINGUISH: Pinch over flame
        if pattern == HandPattern.EXTINGUISH:
            dist = math.hypot(hand_pos[0] - self.wick_pos[0], hand_pos[1] - self.wick_pos[1])
            if dist < 80: # Close to flame
                self.is_lit = False
        
        # SHIELD: Open palm near flame -> Protect (Calm/Grow check)
        # For now, visual feedback: Flame grows slightly to show it's "safe" or shielded
        if pattern == HandPattern.SHIELD:
            dist = math.hypot(hand_pos[0] - self.wick_pos[0], hand_pos[1] - self.wick_pos[1])
            if dist < 150:
                # Effect: Maybe temporarily increase glow or keep stable
                self.glow_radius += 10 # Visual flare

    def draw(self, surface):
        cx, cy = self.center
        
        # Draw Glow if lit
        if self.is_lit:
            # Pulsing glow
            self.glow_radius = 200 + int(math.sin(pygame.time.get_ticks() * 0.005) * 15)
            glow_surf = pygame.Surface((self.glow_radius*2, self.glow_radius*2), pygame.SRCALPHA)
            for r in range(self.glow_radius, 0, -5):
                alpha = int((1 - r/self.glow_radius) * 50)
                pygame.draw.circle(glow_surf, (255, 180, 50, alpha), (self.glow_radius, self.glow_radius), r)
            surface.blit(glow_surf, (self.wick_pos[0] - self.glow_radius, self.wick_pos[1] - self.glow_radius - 30))

        # --- Traditional Clay Diya Design ---
        
        # 1. Base/Pedestal
        pygame.draw.ellipse(surface, (60, 40, 20), (cx - 40, cy + 35, 80, 25)) # Bottom Base
        pygame.draw.rect(surface, (60, 40, 20), (cx - 20, cy + 20, 40, 25)) # Stand
        
        # 2. Base Shadow
        pygame.draw.ellipse(surface, (20, 10, 10), (cx - 70, cy + 20, 140, 30))

        # 3. Main Body (Dark Clay)
        # Using the geometric shapes from "first diagram"
        CLAY_DARK = (101, 67, 33)
        pygame.draw.ellipse(surface, CLAY_DARK, (cx - 70, cy - 30, 140, 70)) 
        
        # 4. The "Beak" (Wick holder part)
        beak_poly = [
            (cx + 30, cy - 30),
            (cx + 100, cy - 20), # Tip extended
            (cx + 40, cy + 30)
        ]
        pygame.draw.polygon(surface, CLAY_DARK, beak_poly)
        
        # 5. Inner Rim / Oil Pool
        pygame.draw.ellipse(surface, (60, 30, 10), (cx - 60, cy - 25, 120, 60)) # Inner Darker
        
        # Oil
        pygame.draw.ellipse(surface, (200, 150, 50), (cx - 50, cy - 20, 100, 50)) # Oil Yellow-ish
        
        # Decor: Gold Trimmings
        # Rim
        pygame.draw.ellipse(surface, GOLD, (cx - 70, cy - 30, 140, 70), 3)
        # Beak Outline
        pygame.draw.lines(surface, GOLD, False, beak_poly, 3)

        # 6. Cotton Wick
        # Line from oil to beak tip
        wick_start = (cx + 20, cy)
        wick_end = self.wick_pos
        pygame.draw.line(surface, (230, 230, 230), wick_start, wick_end, 6)
        
        # Flame
        if self.is_lit:
            self.flame_system.emit(self.wick_pos[0], self.wick_pos[1] - 10, count=5, is_diya_flame=True)
            self.flame_system.update()
            self.flame_system.draw(surface)

def draw_radial_gradient(surface, rect, c1, c2):
    if not hasattr(draw_radial_gradient, 'cache') or draw_radial_gradient.size != rect.size:
        w, h = rect.size
        cw, ch = w // 2, h // 2
        gradient_surf = pygame.Surface((w, h))
        max_dist = math.hypot(cw, ch) * 0.8
        gradient_surf.fill(c2)
        step = 5
        for r in range(int(max_dist), 0, -step):
            ratio = r / max_dist
            if ratio > 1: ratio = 1
            color = (
                int(c1[0] * (1-ratio) + c2[0] * ratio),
                int(c1[1] * (1-ratio) + c2[1] * ratio),
                int(c1[2] * (1-ratio) + c2[2] * ratio)
            )
            pygame.draw.circle(gradient_surf, color, (cw, ch), r)
        draw_radial_gradient.cache = gradient_surf
        draw_radial_gradient.size = rect.size
    surface.blit(draw_radial_gradient.cache, rect)

def main():
    pygame.init()
    pygame.font.init()
    
    # Fullscreen Setup
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    pygame.display.set_caption("Virtual Diya - Realism")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    clock = pygame.time.Clock()
    
    diya = Diya(W, H)
    matchstick = Matchstick()
    stop_btn = StopButton(W - 100, 50)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            # Key Handling
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_q: running = False
                 if event.key == pygame.K_ESCAPE: 
                     diya.reset()
                     matchstick.reset()
            
            # Stop Button Reset
            if stop_btn.handle_event(event): 
                diya.reset()
                matchstick.reset()

        ret, frame = cap.read()
        if not ret:
             frame = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # 1. Background (Gradient)
        draw_radial_gradient(screen, screen.get_rect(), (60, 10, 20), BLACK) 
        
        # 2. Matchstick Logic (Thumb + Middle Finger Rub) - LEFT HAND ONLY
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Check Handedness
                label = "Unknown"
                if results.multi_handedness:
                    # In a mirrored (flipped) image, Physical Right Hand is labeled "Left" by MediaPipe
                    label = results.multi_handedness[idx].classification[0].label
                
                # We want Physical Left Hand -> Label "Right" (Mirrored)
                if label != "Right": 
                    continue 

                # Use this hand
                # Pattern Recognition & Matchstick Update
                # We pass the full landmarks list normalized, and W/H for scaling inside
                matchstick.update(hand_landmarks.landmark, W, H)
                
                # Diya Interactions (Shield/Extinguish)
                # Need a representative position for the hand (e.g., Index Base or Center)
                idx_base = hand_landmarks.landmark[5]
                hand_px = (int(idx_base.x * W), int(idx_base.y * H))
                
                # Detect pattern specifically for Diya (might be same hand or different if we enabled two hands)
                # For now using same hand
                current_pattern = matchstick.pattern_recognizer.detect(hand_landmarks.landmark)
                diya.handle_interaction(current_pattern, hand_px)
                
                break # Only use the first valid Right hand
        
        matchstick.draw(screen)
        
        # 3. Diya
        diya.check_ignition(matchstick.head_pos, matchstick.is_lit)
        diya.draw(screen)
        
        # 4. Stop Button
        stop_btn.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
