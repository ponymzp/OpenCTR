"""
计算效果
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def cal_all(y_pred, y):
    y_binary = (y_pred > 0.5).astype(int)

    # 计算准确率
    accuracy = accuracy_score(y, y_binary)
    print(f"Accuracy: {accuracy:.4f}")

    # 计算精确率
    precision = precision_score(y, y_binary)
    print(f"Precision: {precision:.4f}")

    # 计算召回率
    recall = recall_score(y, y_binary)
    print(f"Recall: {recall:.4f}")

    # 计算 F1 分数
    f1 = f1_score(y, y_binary)
    print(f"F1 Score: {f1:.4f}")


def model_train(model, X_train, y_train):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    model.fit(
        X_train, y_train,
        validation_split=0.3, shuffle=True, batch_size=32, epochs=1, verbose=2
    )
    model.summary()


def model_predict(model, X_test, y_test):
    # 模型预测
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cal_all(y_pred, y_test)
