from fastapi import FastAPI
import time

app = FastAPI()

@app.get("/work")
def work():
    # CPU를 조금 쓰게 하는 간단한 연산
    total = 0
    for i in range(300000):
        total += i * i

    return {"status": "ok", "value": total}