import joblib # you may use native pickle here as well

#TO RUN: python /output/boxing/readplk.py

output = joblib.load('output/boxing/vibe_output.pkl') 

print(output.keys())  

for k,v in output[1].items(): print(k,v.shape)