import json

if __name__ == "__main__":

    with open('data/val/test.json', 'r') as f:
        test = json.loads(f.read())

    print(test[0]["question_id"])