# Hand Patterns Verification Walkthrough

This guide explains how to verify the newly added hand patterns for the Virtual Diya project.

## 1. Setup
Ensure you are in the correct environment and have a webcam connected. Run the Diya application:

```bash
python src/ui/diya.py
```

## 2. Testing Hand Patterns

### A. Matchstick Holding (Default)
- **Gesture**: Keep your **Index finger extended** (others can be curled).
- **Expected Result**: A matchstick should appear attached to your index finger tip, following your movement.

### B. Ignite Key
- **Gesture**: **Rub or Snap** your **Thumb and Middle finger** together while holding the match (Index extended).
- **Expected Result**: The matchstick head should catch fire.

### C. Lighting the Diya
- **Action**: Bring the **lit matchstick head** close to the Diya's wick (center of the lamp).
- **Expected Result**: The Diya should light up with a flame and glow.

### D. Shielding (Protecting the Flame)
- **Gesture**: **Open your palm completely** (all fingers extended and close together) and bring it near the Diya flame.
- **Expected Result**: The Diya's glow radius should increase, simulating the reflection of light on your hand or the flame calming down.

### E. Extinguishing
- **Gesture**: Make a **"Pinch" gesture** (Thumb and Index finger touching) directly over the Diya flame.
- **Expected Result**: The Diya flame should extinguish immediately.

## 3. Troubleshooting
- **Lighting**: Ensure your hand is well-lit for MediaPipe tracking.
- **Detection**: If patterns aren't detected, try moving your hand closer or further from the camera.
- **Handedness**: The system is configured for the **Physical Left Hand** (which appears as the **Right Hand** in the mirrored camera view).
