import numpy as np
import pandas as pd

import os,sys

def textprocessingOriginal(doc):
  # Remove numbers in product name
  doc = doc.str.replace("[^a-zA-Z]", " ")

  return doc