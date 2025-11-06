"""
Policy Network Quantization for Teensy Deployment
- INT8 quantization for efficient on-board inference
- Weight packing and serialization
- CRC32 checksum for safe transfer
"""

import torch
import torch.nn as nn
import numpy as np
import struct
import zlib
from typing import Tuple, List, Dict
from pathlib import Path


class QuantizedPolicy:
    """
    Quantized policy network for deployment to Teensy

    Quantization scheme:
    - Weights: INT8 [-127, 127] with per-layer scale factors
    - Activations: INT16 during inference for precision
    - Logits output: INT16, converted to float for argmax on Teensy
    """

    def __init__(self, policy_network: nn.Module):
        """
        Args:
            policy_network: Trained PolicyNetwork to quantize
        """
        self.policy_network = policy_network
        self.policy_network.eval()

        self.quantized_weights = {}
        self.scale_factors = {}
        self.zero_points = {}

    def quantize(self):
        """Quantize all layers of the policy network"""
        with torch.no_grad():
            for name, param in self.policy_network.named_parameters():
                if 'weight' in name or 'bias' in name:
                    self._quantize_tensor(name, param.cpu().numpy())

    def _quantize_tensor(self, name: str, tensor: np.ndarray):
        """
        Quantize single tensor to INT8

        Symmetric quantization: Q = round(R / S)
        Where S = max(abs(R)) / 127
        """
        # Calculate scale factor
        max_val = np.max(np.abs(tensor))
        if max_val == 0:
            max_val = 1.0  # Avoid division by zero

        scale = max_val / 127.0
        self.scale_factors[name] = scale

        # Quantize
        quantized = np.round(tensor / scale).astype(np.int8)
        self.quantized_weights[name] = quantized

        # Zero point (for symmetric quantization, always 0)
        self.zero_points[name] = 0

    def serialize(self) -> bytes:
        """
        Serialize quantized weights to binary format

        Format:
        - Header: [num_layers (uint16)]
        - For each layer:
          - name_len (uint8), name (str), shape (uint16[]), scale (float32), data (int8[])
        """
        buffer = bytearray()

        # Number of layers
        num_layers = len(self.quantized_weights)
        buffer.extend(struct.pack('<H', num_layers))

        # Serialize each layer
        for name, weights in self.quantized_weights.items():
            # Name
            name_bytes = name.encode('utf-8')
            buffer.extend(struct.pack('<B', len(name_bytes)))
            buffer.extend(name_bytes)

            # Shape
            shape = weights.shape
            buffer.extend(struct.pack('<H', len(shape)))
            for dim in shape:
                buffer.extend(struct.pack('<H', dim))

            # Scale factor
            scale = self.scale_factors[name]
            buffer.extend(struct.pack('<f', scale))

            # Quantized data
            buffer.extend(weights.tobytes())

        return bytes(buffer)

    def save(self, path: str):
        """Save quantized policy to file"""
        serialized = self.serialize()

        with open(path, 'wb') as f:
            f.write(serialized)

        print(f"Saved quantized policy to {path}")
        print(f"Size: {len(serialized)} bytes ({len(serialized)/1024:.2f} KB)")

    @staticmethod
    def load(path: str) -> Tuple[Dict, Dict]:
        """Load quantized policy from file"""
        with open(path, 'rb') as f:
            data = f.read()

        weights = {}
        scales = {}

        offset = 0

        # Read number of layers
        num_layers, = struct.unpack('<H', data[offset:offset+2])
        offset += 2

        # Read each layer
        for _ in range(num_layers):
            # Name
            name_len, = struct.unpack('<B', data[offset:offset+1])
            offset += 1
            name = data[offset:offset+name_len].decode('utf-8')
            offset += name_len

            # Shape
            ndims, = struct.unpack('<H', data[offset:offset+2])
            offset += 2
            shape = []
            for _ in range(ndims):
                dim, = struct.unpack('<H', data[offset:offset+2])
                offset += 2
                shape.append(dim)

            # Scale
            scale, = struct.unpack('<f', data[offset:offset+4])
            offset += 4
            scales[name] = scale

            # Data
            size = np.prod(shape)
            weight_data = np.frombuffer(data[offset:offset+size], dtype=np.int8)
            offset += size
            weights[name] = weight_data.reshape(shape)

        return weights, scales

    def chunk_for_transfer(self, chunk_size: int = 256) -> List[Tuple[int, bytes, int]]:
        """
        Split serialized policy into chunks for ROS transfer

        Returns:
            List of (chunk_index, chunk_data, crc32)
        """
        serialized = self.serialize()
        total_size = len(serialized)
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)
            chunk_data = serialized[start:end]

            # Calculate CRC32 checksum
            crc = zlib.crc32(chunk_data) & 0xFFFFFFFF

            chunks.append((i, chunk_data, crc))

        return chunks


def test_quantization():
    """Test quantization pipeline"""
    from sac_discrete import PolicyNetwork

    # Create dummy policy
    obs_dim = 50
    num_actions = 16
    policy = PolicyNetwork(obs_dim, num_actions, hidden_dim=64)

    # Quantize
    quant_policy = QuantizedPolicy(policy)
    quant_policy.quantize()

    # Serialize
    serialized = quant_policy.serialize()
    print(f"Serialized size: {len(serialized)} bytes")

    # Save
    quant_policy.save("test_policy.bin")

    # Load
    weights, scales = QuantizedPolicy.load("test_policy.bin")
    print(f"Loaded {len(weights)} layers")

    # Test chunking
    chunks = quant_policy.chunk_for_transfer(chunk_size=256)
    print(f"Split into {len(chunks)} chunks")

    # Verify dequantization error
    with torch.no_grad():
        test_input = torch.randn(1, obs_dim)
        original_output = policy(test_input)

        # Simulate dequantization
        reconstructed_fc1_weight = torch.from_numpy(
            weights['fc1.weight'].astype(np.float32) * scales['fc1.weight']
        )

        # Calculate relative error
        original_weight = policy.fc1.weight.data.cpu().numpy()
        quantized_weight = weights['fc1.weight'].astype(np.float32) * scales['fc1.weight']
        rel_error = np.mean(np.abs(original_weight - quantized_weight) / (np.abs(original_weight) + 1e-8))

        print(f"Quantization relative error: {rel_error:.6f}")


class TeensyPolicyInference:
    """
    Simulated Teensy inference engine (for testing on PC)
    Mimics INT8 operations that will run on Teensy
    """

    def __init__(self, weights: Dict, scales: Dict):
        self.weights = weights
        self.scales = scales

    def infer(self, obs: np.ndarray) -> int:
        """
        Simulate INT8 inference

        Args:
            obs: Normalized observation (float32)

        Returns:
            action_id: Discrete action (0-15)
        """
        # Quantize input to INT16
        obs_scale = np.max(np.abs(obs)) / 32767.0
        if obs_scale == 0:
            obs_scale = 1.0
        obs_q = np.round(obs / obs_scale).astype(np.int16)

        # Layer 1: fc1
        w1 = self.weights['fc1.weight']
        b1 = self.weights['fc1.bias']
        s1 = self.scales['fc1.weight']
        sb1 = self.scales['fc1.bias']

        # Matrix multiply (INT8 x INT16 = INT32, then scale)
        h1 = np.dot(w1, obs_q).astype(np.int32)
        h1 = (h1 * obs_scale * s1).astype(np.int16)
        h1 += (b1 * sb1).astype(np.int16)

        # ReLU
        h1 = np.maximum(h1, 0)

        # Layer 2: fc2
        w2 = self.weights['fc2.weight']
        b2 = self.weights['fc2.bias']
        s2 = self.scales['fc2.weight']
        sb2 = self.scales['fc2.bias']

        # Scale h1 for next layer
        h1_scale = np.max(np.abs(h1)) / 32767.0
        if h1_scale == 0:
            h1_scale = 1.0
        h1_norm = (h1 / h1_scale).astype(np.int16)

        h2 = np.dot(w2, h1_norm).astype(np.int32)
        h2 = (h2 * h1_scale * s2).astype(np.int16)
        h2 += (b2 * sb2).astype(np.int16)
        h2 = np.maximum(h2, 0)

        # Layer 3: logits_out
        w3 = self.weights['logits_out.weight']
        b3 = self.weights['logits_out.bias']
        s3 = self.scales['logits_out.weight']
        sb3 = self.scales['logits_out.bias']

        h2_scale = np.max(np.abs(h2)) / 32767.0
        if h2_scale == 0:
            h2_scale = 1.0
        h2_norm = (h2 / h2_scale).astype(np.int16)

        logits = np.dot(w3, h2_norm).astype(np.int32)
        logits = (logits * h2_scale * s3).astype(np.int16)
        logits += (b3 * sb3).astype(np.int16)

        # Argmax (no need for softmax on Teensy)
        action = np.argmax(logits)

        return int(action)


if __name__ == '__main__':
    test_quantization()
