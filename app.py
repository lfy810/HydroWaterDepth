from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from main import main
from main1 import IMAGE_PATH

app = FastAPI()

# ------ 跨域支持 START ------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 本地开发允许所有，生产建议写死前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------ 跨域支持 END ------

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 这里 IMAGE_PATH 的作用方式要注意，如果 main1.py 只是定义常量会有问题
    # 更稳妥写法见下行注释
    # main1.IMAGE_PATH = temp_path   # 如果 main1.py 定义为变量可以用
    # 或者 main() 内部读取 temp_path 变量

    try:
        # 如果 main() 里用 IMAGE_PATH，可以用上面方式传参；也可以改成 main(temp_path)
        depth_cm = main()
        return JSONResponse({
            "status": "success",
            "depth_cm": round(depth_cm, 1)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
