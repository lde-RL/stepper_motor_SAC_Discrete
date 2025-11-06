# Action command message
# Policy → Teensy control

uint8 action_mask  # 4-bit switch mask (b3 b2 b1 b0)
                   # Illegal combinations will be masked by firmware safety guard
