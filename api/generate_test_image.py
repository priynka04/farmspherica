"""
Simple script to generate a test image for API testing.
Run this and it creates: test.jpg in your project folder
"""

from PIL import Image, ImageDraw
import random

# Create a new image (640x640, RGB)
img = Image.new('RGB', (640, 640), color='white')
draw = ImageDraw.Draw(img)

# Draw a green background (like plant leaves)
draw.rectangle([(0, 0), (640, 640)], fill=(34, 139, 34))

# Draw some random shapes to simulate plant parts
for _ in range(5):
    x1 = random.randint(50, 300)
    y1 = random.randint(50, 300)
    x2 = x1 + random.randint(100, 250)
    y2 = y1 + random.randint(100, 250)
    color = (random.randint(0, 100), random.randint(100, 200), random.randint(0, 100))
    draw.ellipse([(x1, y1), (x2, y2)], fill=color)

# Draw some brown stems
for _ in range(3):
    x1 = random.randint(100, 400)
    y1 = random.randint(200, 500)
    x2 = x1 + random.randint(20, 60)
    y2 = y1 + random.randint(100, 200)
    draw.line([(x1, y1), (x2, y2)], fill=(139, 69, 19), width=5)

# Add some "disease spots" (red circles to simulate disease)
for _ in range(3):
    x = random.randint(100, 500)
    y = random.randint(100, 500)
    size = random.randint(10, 40)
    draw.ellipse([(x, y), (x + size, y + size)], fill=(255, 0, 0))

# Save the image
img.save('test.jpg')
print("✅ Test image created: test.jpg")
print("   You can now test the API upload with this image")