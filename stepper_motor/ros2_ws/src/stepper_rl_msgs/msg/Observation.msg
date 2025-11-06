# Observation message (NO encoder in observation!)
# Teensy → PC

# --- Target information (for policy input) ---
float32 theta_ref_now         # Current target angle (normalized)
float32 dtheta_ref_now        # Current target velocity (0 for static)
float32[] theta_ref_preview   # Future target angles (K elements)

# --- Action history (encoder substitute hint) ---
uint8 last_action_mask        # Previous 4-bit switch state (e.g., 0b0101)
uint8[] action_hist           # Recent N-step action history (e.g., N=16)

# --- Episode/timing info ---
uint32 step_count             # Current step in episode
float32 elapsed_time          # Time since episode start (sec)

# --- Safety flags (for reward/termination only) ---
uint8 fault_flags             # Bit flags: overcurrent, overheat, comm timeout, etc.
