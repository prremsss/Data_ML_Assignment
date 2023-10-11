import uvicorn
import sys
sys.path.append('C:/Users/LEGION/Desktop/MASTER/PFE/web/Data_ML_Assignment')
from  src.api.server import server

if __name__ == "__main__":
    serving_app = server()
    uvicorn.run(
        serving_app,
        host="0.0.0.0",
        port=9000,
        log_level="info",
    )