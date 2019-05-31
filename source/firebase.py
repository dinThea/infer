import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

class Firebase:
    
    def __init__(self):

        self.cred = credentials.Certificate('vtex-hackthon-7f88a532fac6.json') 
        firebase_admin.initialize_app(self.cred, options={
            'databaseURL': 'https://vtex-hackthon.firebaseio.com/',
        })   
        self.db = db.reference('store-data/data/')
    
    def create_data(self, json_data):

        self.db.set({'bbs':json_data['bbs'],
                     'labels':json_data['labels'],
                     'last_label':json_data['labels'][0]
                     })
