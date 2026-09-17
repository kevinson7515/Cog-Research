import base64
import hashlib

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


class CipherUtils:
    @staticmethod
    def encrypt(text: str) -> str:
        seed_key, seed_iv = CipherUtils.generate_key_iv()
        cipher = AES.new(seed_key, AES.MODE_CBC, seed_iv)
        padded_text = pad(text.encode(), AES.block_size)
        encrypted = cipher.encrypt(padded_text)
        return base64.b64encode(encrypted).decode('utf-8')

    @staticmethod
    def decrypt(data: str) -> str:
        encrypted = base64.b64decode(data)
        seed_key, seed_iv = CipherUtils.generate_key_iv()
        cipher = AES.new(seed_key, AES.MODE_CBC, seed_iv)
        padded_text = cipher.decrypt(encrypted)
        return unpad(padded_text, AES.block_size).decode('utf-8')

    @staticmethod
    def generate_key_iv():
        # Create a SHA-256 hash of the seed string
        hash_bytes = hashlib.sha256("aim_traffic_ops_2024".encode()).digest()
        # Use the first 16 bytes for the key (AES-128)
        key = hash_bytes[:16]
        # Use the next 16 bytes for the IV
        iv = hash_bytes[16:32]
        return key, iv


# Example usage
if __name__ == "__main__":
    plain_text = "traffic_ops_token_key"

    encrypted_text = CipherUtils.encrypt(plain_text)
    # print(f"Encrypted: {encrypted_text}")

    decrypted_text = CipherUtils.decrypt(encrypted_text)
    # print(f"Decrypted: {decrypted_text}")
