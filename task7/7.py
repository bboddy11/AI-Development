import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RecommendationSystem:
    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings_df = ratings_df.copy()
        self.user_item_matrix = None
        self.similarity_matrix = None
        self._prepare_data()

    def _prepare_data(self):
        logging.info("Формування матриці користувач-товар...")
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='User', columns='Item', values='Rating'
        ).fillna(0)

        logging.info("Обчислення косинусної схожості...")
        similarity = cosine_similarity(self.user_item_matrix)
        self.similarity_matrix = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        logging.info("Матриця схожості побудована.")

    def recommend(self, target_user: str, top_n: int = 3):
        if target_user not in self.similarity_matrix.index:
            logging.warning(f"Користувач '{target_user}' не знайдений у системі.")
            return pd.Series()

        logging.info(f"Формування рекомендацій для користувача '{target_user}'...")

        # Отримання схожих користувачів
        similar_users = self.similarity_matrix[target_user].drop(index=target_user)
        similar_users = similar_users[similar_users > 0].sort_values(ascending=False)

        # Зважене об'єднання оцінок товарів
        weighted_scores = pd.Series(dtype=np.float64)
        for user, similarity in similar_users.items():
            weighted_scores = weighted_scores.add(
                self.user_item_matrix.loc[user] * similarity, fill_value=0
            )

        # Вилучення товарів, які вже були оцінені цільовим користувачем
        seen_items = self.user_item_matrix.loc[target_user]
        seen_items = seen_items[seen_items > 0].index
        recommendations = weighted_scores.drop(index=seen_items, errors='ignore')

        # Нормалізація та повернення топ-N
        recommendations = recommendations.sort_values(ascending=False).head(top_n)
        logging.info(f"Знайдено {len(recommendations)} рекомендацій.")

        return recommendations

    def show_similarity_matrix(self):
        print("\nМатриця схожості користувачів:")
        print(self.similarity_matrix.round(2).to_string())


# Створення штучного набору даних
data = {
    'User': ['User1', 'User1', 'User1', 'User2', 'User2', 'User3', 'User3', 'User4', 'User4', 'User5'],
    'Item': ['Item1', 'Item2', 'Item3', 'Item1', 'Item3', 'Item2', 'Item4', 'Item1', 'Item4', 'Item3'],
    'Rating': [5, 3, 2, 4, 5, 4, 4, 2, 5, 4]
}
df_ratings = pd.DataFrame(data)

# Ініціалізація системи та генерація рекомендацій
recommender = RecommendationSystem(df_ratings)
recommender.show_similarity_matrix()

user_to_recommend = "User1"
recommendations = recommender.recommend(user_to_recommend, top_n=3)

print(f"\n📌 Рекомендовані товари для {user_to_recommend}:\n")
if not recommendations.empty:
    for item, score in recommendations.items():
        print(f"🛒 {item}: Рекомендаційний бал = {score:.2f}")
else:
    print("Немає доступних рекомендацій.")
