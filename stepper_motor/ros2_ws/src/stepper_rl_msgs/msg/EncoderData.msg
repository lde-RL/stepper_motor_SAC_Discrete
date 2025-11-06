# Encoder data message (TRAINING ONLY - for reward calculation)
# Teensy → PC (only during training, removed after deployment)

float32 theta_mech     # Mechanical angle (rad)
float32 theta_elec     # Electrical angle (rad)
float32 rpm            # Mechanical RPM
int32 raw_count        # Raw encoder count
