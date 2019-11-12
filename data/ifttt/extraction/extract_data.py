"""
This script extracts IFTTT recipes from the MSR IFTTT curated dataset.
"""
#%%
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import pickle
import pandas as pd 
from collections import Counter
import numpy as np
from tqdm import tqdm

# all_urls = "data/ifttt/extraction/data/all.urls"
# bad_urls = []
# with open(all_urls, "r") as all_urls:
#     all_urls = all_urls.readlines()

# all_urls = [url.strip() for url in all_urls][86000:]
# url_len = len(all_urls)
# for i, url in tqdm(enumerate(all_urls), total=url_len):

#     try:
#         r = requests.get(url + ".html")
#         soup = BeautifulSoup(r.text, 'html.parser')
#         title = soup.title.text.strip()
        
#         try:
#             desc = soup.findAll("meta", {"name": "description"})[0]["content"]

#         except:
#             desc = ""
        
#         url_map[url] = {"title": title, "description": desc}

#     except:
#         url_map[url] = {}
#         bad_urls.append(url)
#         print("excepting: ", url)

# pickle.dump(url_map, open("url_map_complete.pkl", "wb"))

# Develop new train, val, test set
# %%

# Rescraped data with original titles
url_map = pickle.load(open("url_map_complete.pkl", "rb"))

#%%
# turk annotated labels for test set
# with open("data/ifttt/extraction/data/test.urls", "r") as f:
#     test_urls = f.readlines()

# test_urls = [url.strip() for url in test_urls]
# turk = pd.read_csv("data/ifttt/extraction/data/turk_public.tsv", sep='\t')
# turk.head()


#%%
# def manual_parse(text):
#     if text == "" or len(text.split(":")) != 3:
#         return {}
#     p = text.split(":")
#     p2 = p[1].split(".")
#     return {"trigger_channel": p[0].strip().lower().replace(" ", "_"),
#             "trigger_function": p2[0].strip().lower().replace(" ", "_"),
#             "action_channel": p2[1].strip().lower().replace(" ", "_"),
#             "action_fuction:": p[2].strip().lower().replace(" ", "_")}


#%%
# test_clean = []
# for i,g in turk.groupby("URL"):

#     text = url_map[i]["title"].replace("- IFTTT", "")
#     if text == "IFTTT / 404 Error" or len(text.split()) < 3:
#         continue

#     text = " ".join([t.strip() for t in text.split()])
#     text = text.replace("- IFTTT", "").replace('"', "").strip().lower()
#     desc = url_map[i]["description"]


#     # Extract turker annotations 
#     triggers = [k.strip().lower().replace(" ","_") for k in g["Trigger channel"]]
#     tfunc = [k.strip().lower().replace(" ", "_") for k in g["Trigger"]]
#     actions = [k.strip().lower().replace(" ","_") for k in g["Action channel"]]
#     afunc = [k.strip().lower().replace(" ", "_") for k in g["Action"]]

#     all_vals = [triggers, tfunc, actions, afunc]

#     skip_flag = False 
#     seq = []   
#     for vals in all_vals:
#         most_common = Counter(vals).most_common(1)
#         if most_common[0][1] < 3:
#             skip_flag = True
#             break
#         else:
#             seq.append(most_common[0][0].strip().lower().replace(" ","_"))
    
#     if skip_flag or 'unintelligible' in seq or 'nonenglish' in seq or len(seq) == 0:       
#         continue
#     else:
#         test_clean.append({"text": text,
#                            "url": i,
#                            "desc": desc,
#                            "target": " ".join(seq),
#                            "seq": seq})
# print("finished")


# %%
# df = pd.DataFrame(test_clean)
# df.to_csv("data/ifttt/test_clean.csv", index=False)
# print(len(test_clean))

#%%
# with open("data/ifttt/src_test.txt", "w") as src, \
#      open("data/ifttt/target_test.txt", "w") as target:
#      for i,v in df.iterrows():
#         src.write(v.text + " \n")
#         target.write(v.target + " \n")

# # %%
# with open("data/ifttt/extraction/data/dev.urls") as dev_urls, \
#      open("data/ifttt/extraction/data/train.urls") as train_urls:
#      dev_urls = dev_urls.readlines()
#      train_urls = train_urls.readlines()

# dev_urls = [url.strip() for url in dev_urls]
# train_urls = [url.strip() for url in train_urls]



# # %%
# la_dat = pickle.load(open("data/ifttt/msr_data_py3.pkl", "rb"))

# #%%
# la_dat.keys()


# # %%
# la_dev   = la_dat["dev"]
# la_train = la_dat["train"]

# # %%
# print(len(la_dev), len(dev_urls))
# print(len(la_train), len(train_urls))

# #%%
# def isEnglish(s):
#     try:
#         s.encode(encoding='utf-8').decode('ascii')
#     except UnicodeDecodeError:
#         return False
#     else:
#         return True

# def check_input(text: str) -> bool:
#     if not isEnglish(text):
#         return False

#     text = text.replace("- IFTTT", "")
#     if text == "IFTTT / 404 Error" or len(text.split()) < 3:
#         return False

#     return True

# # # %%
# # dev = []
# # for i in la_dev:
# #     url = i["url"]
# #     labels = i["label_names"]
# #     text = url_map[url]["title"]
# #     desc = url_map[url]["description"]

# #     if not check_input(text):
# #         continue

# #     text = " ".join([t.strip() for t in text.split()])
# #     text = text.replace("- IFTTT", "").replace('"', "").strip().lower()

# #     labels = [lab.decode('UTF-8').strip().lower() for lab in labels]
# #     target = " ".join(labels)

# #     dev.append({"text": text,
# #                         "url": url,
# #                         "desc": desc,
# #                         "target": target,
# #                         "seq": labels})

# # dev_df = pd.DataFrame(dev)
# # dev_df.head()

# # # %%
# # dev_df.to_csv("data/ifttt/dev_clean.csv", index=False)


# # # %%
# # with open("data/ifttt/src_val.txt", "w") as f:
# #     src = [s + " \n" for s in dev_df.text.tolist()]
# #     f.writelines(src)

# # with open("data/ifttt/target_val.txt", "w") as f:
# #     target = [s + " \n" for s in dev_df.target.tolist()]
# #     f.writelines(target)


# # %%
# train = []
# bc = []

# for i in tqdm(la_train):
#     url = i["url"]
#     labels = i["label_names"]
#     text = url_map[url]["title"]
#     desc = url_map[url]["description"]

#     if not check_input(text):
#         bc.append(text)
#         continue

#     text = " ".join([t.strip() for t in text.split()])
#     text = text.replace("- IFTTT", "").replace('"', "").strip().lower()

#     labels = [lab.decode('UTF-8').strip().lower() for lab in labels]
#     target = " ".join(labels)

#     train.append({"text": text, "url": url, "desc": desc, "target": target,
#                   "seq": labels})

# train_df = pd.DataFrame(train)
# train_df.head()

# print("bad ", len(bc))
# train_df.to_csv("data/ifttt/train_clean.csv", index=False)

# ## %%
# with open("data/ifttt/src_train.txt", "w") as f:
#     src = [s + " \n" for s in train_df.text.tolist()]
#     f.writelines(src)

# with open("data/ifttt/target_train.txt", "w") as f:
#     target = [s + " \n" for s in train_df.target.tolist()]
#     f.writelines(target)

# with open("bad_train.txt","w") as f:
#     for b in bc:
#         f.write(b + " \n")

# Regenerate test set. Use LA Attention paper value but our filtered ids
#%%
test_clean = pd.read_csv("data/ifttt/test_clean.csv")
la = pickle.load(open("data/ifttt/msr_data_py3.pkl", "rb"))


# %%
la_test = la["test"]
la_test_map = {m["url"]: m for m in la_test}
# %%
test_v1 = [] # relabel functions with channel prepended
test_v2 = [] # used functions from lattent attention paper

skipped = []
for i, v in test_clean.iterrows():
    
    # Relabel base labels
    bl = v.target.split()
    bl[1] = bl[0] + "." + bl[1]
    bl[3] = bl[2] + "." + bl[3]
    bl_target = " ".join(bl)
    test_v1.append({"src": v.text, "target": bl_target})

    # Grab label from LA paper dataset
    try:
        url = v.url
        la_labels = la_test_map[url]["label_names"]
        la_target = " ".join([lab.decode('UTF-8').strip().lower().replace(" ","_") for lab in la_labels])

        test_v2.append({"src": v.text, "target": la_target})
    except:
        skipped.append(url)
        continue


# %%
print(len(test_v1), len(test_v2))

# %%
with open("data/ifttt/src_test_v1.txt", "w") as src_file, \
     open("data/ifttt/target_test_v1.txt", "w") as target_file:
     src = [s["src"]+" /n" for s in test_v1]
     src_file.writelines(src)
     target = [s["src"]+" /n" for s in test_v1]
     target_file.writelines(target)

# %%
with open("data/ifttt/src_test_v2.txt", "w") as src_file, \
     open("data/ifttt/target_test_v2.txt", "w") as target_file:
     src = [s["src"]+" /n" for s in test_v2]
     src_file.writelines(src)
     target = [s["src"]+" /n" for s in test_v2]
     target_file.writelines(target)

# %%
