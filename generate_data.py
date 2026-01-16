import pandas as pd
import random

data = []
for _ in range(50):
    followers = random.randint(10, 100000)
    following = random.randint(10, 5000)
    posts = random.randint(0, 1000)
    is_private = random.choice([0, 1])
    
    # Simple rule: if followers are low and following is high, it's likely fake
    label = 1 if followers < 100 and following > 1000 else 0
    data.append([followers, following, posts, is_private, label])

df = pd.DataFrame(data, columns=["followers", "following", "posts", "is_private", "label"])
df.to_csv("instagram_data.csv", index=False)
