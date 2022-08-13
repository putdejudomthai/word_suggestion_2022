from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from word_suggestions import Suggestions
#======================================================#

origins = ['*']
app = FastAPI()
app.add_middleware(CORSMiddleware,
                  allow_origins = origins,
                  allow_credentials=True,
                  allow_methods=['*'],
                  allow_headers=['*'])
suggest = Suggestions()
#======================================================#

class WordSuggestInput(BaseModel) :
  payload : str

#======================================================#

@app.post('/suggest')
async def suggest_render(data : WordSuggestInput) :
  return {
          "ori_text" : data.payload,
          "sug" : suggest.word_suggest(data.payload)
          }