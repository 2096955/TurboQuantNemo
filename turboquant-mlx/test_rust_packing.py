import numpy as np
import turboquant_storage

x = np.random.randint(0, 8, size=(100,), dtype=np.uint8)
packed = turboquant_storage.pack_3bit(x)
unpacked = turboquant_storage.unpack_3bit(packed, len(x))

if np.array_equal(x, unpacked):
    print("Rust packed storage works perfectly!")
else:
    print("Mismatch!")
    print("Expected:", x[:16])
    print("Got:     ", unpacked[:16])