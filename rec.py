# create_mock_model.py
import pickle
import pandas as pd

class MockRecommender:
    def __init__(self, df):
        self.df = df

    def recommend(self, user_id, top_n=5):
        # 随便推荐5个热门游戏（只是为了测试界面）
        top_games = (
            self.df["game-title"]
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )
        return top_games
    
df = pd.DataFrame({
    "user-id": [1,1,2,3],
    "game-title": ["Dota2","CS:GO","Dota2","Hades"]
})

with open("model.pkl", "wb") as f:
    pickle.dump(MockRecommender(df), f)
