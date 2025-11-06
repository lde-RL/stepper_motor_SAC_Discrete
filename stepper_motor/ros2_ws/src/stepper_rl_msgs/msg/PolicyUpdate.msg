# Policy weight update message (chunked transfer)
# PC → Teensy

uint32 version         # Weight version number (for rollback/comparison)
uint16 total_chunks    # Total number of chunks
uint16 index           # Current chunk index (0-indexed)
uint32 crc32           # CRC32 checksum for this chunk
uint8[] payload        # Weight data (int8 quantized)
