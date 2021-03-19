from serpapi import GoogleSearch
search = GoogleSearch({"q": "coffee", "location": "Austin,Texas", "api_key": "secretKey"})
result = search.get_dict()
print(result)
