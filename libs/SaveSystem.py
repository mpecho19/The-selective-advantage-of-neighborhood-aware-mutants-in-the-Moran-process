import pickle
import signal
import time
import os
from typing import Any, Union


#TODO: Folder support
class SaveSystem:
    def __init__(self, name : str, load : bool, folder = None ):
        self.file = None
        self.variables = {}
        self.name = name
        self.folder = folder
        if folder:
            if not os.path.exists("SavedFiles/" + folder):
                os.makedirs("SavedFiles/" + folder)
            self.location = f"SavedFiles/{folder}/{name}.pkl"
            self.temp_location = f"SavedFiles/{folder}/{name}.tmp"
        else:
            self.location = "SavedFiles/" +name + ".pkl"
            self.temp_location = "SavedFiles/" + name + ".tmp"
        self.autosave = False
        self.__auto_init_lists = False
        if load:
            self.variables = self.load()
    

    @staticmethod
    def is_filename_free(name):
        return not os.path.exists("SavedFiles/" + name + ".pkl")        

    def auto_init_lists(self, set_bool : bool):
        self.__auto_init_lists = set_bool    
    def set_autosave(self, autosave_bool):
        self.autosave = autosave_bool
    
    #TODO: Folder support
    def rename_file(self, name):
        self.name = name
        new_location = "SavedFiles/" +name + ".pkl"
        os.rename(self.location, new_location)
        self.location = new_location
        self.temp_location = "SavedFiles/" + name + ".tmp"



    def save(self):
        def handler(signum, frame):
            print("Saving in progress. Please wait...")
    
        # Block SIGINT
        original_sigint_handler = signal.signal(signal.SIGINT, handler)
        original_sigterm_handler = signal.signal(signal.SIGTERM, handler)
        
        self.__set_date()
        try:
            with open(self.temp_location, 'wb') as file:
                pickle.dump(self.variables, file)
            if os.path.exists(self.location):
                os.remove(self.location)
            os.rename(self.temp_location, self.location)
        finally:
            # Restore original handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)
        
    def load(self):
        if not os.path.exists(self.location):
            return {}
        with open(self.location, 'rb') as file:
            self.variables = pickle.load(file)
            return self.variables
    
    def clear_data(self):
        self.variables = {}
        if os.path.exists(self.location):
            os.remove(self.location)
    
    def set_description(self, description):
        self.variables["_description"] = description    
        if self.autosave:
            self.save()
            
    
    def init_lists(self, *args):
        for key in args:
            if key in self.variables:
                print("Error: Key already exists")
            else:
                self.variables[key] = []
                    
    def __set_date(self):    
        self.variables["_date"] = time.strftime("%d.%m.%Y %H:%M:%S")
    
    def key_exists(self, key):
        return key in self.variables
    
    def remove_item(self, key):
        if key in self.variables:
            del self.variables[key]
            if self.autosave:
                self.save()
    
    def delete_iteM(self, key):
        self.remove_item(key)
    
    def info(self):
        print("#########################")
        if "_date" in self.variables:
            print("Date:", self.variables["_date"])
        if "_description" in self.variables:
            print("Description:", self.variables["_description"])
        print("#########################")

    
    def print_variables(self, names_only = False):
        print("#########################")
        print()
        
        for key, value in self.variables.items():
                if isinstance(key, str): 
                    if key[0] == "_":
                        continue
                if isinstance(value, list):
                    if names_only:
                        print(f"{key} : (len = {len(value)})")
                    else:
                        print(f"{key} : ({len(value)}) {value}")
                else:
                    if names_only:
                        print(f"{key}")
                    else:
                        print(f"{key} : {value}")
                if(not names_only):    
                    print()
                    
        if(names_only):
            print()
        print("#########################")
        if "_date" in self.variables:
            print("Date:", self.variables["_date"])

        if "_description" in self.variables:
            print("Description:", self.variables["_description"])
        print("#########################")
                    
    def __getitem__(self, key) -> Union[list, any]:
        if key not in self.variables and self.__auto_init_lists:
            self.variables[key] = []
            return self.variables[key]
        return self.variables[key]
    
    def __setitem__(self, key, value):
        if key not in self.variables and self.__auto_init_lists:
            self.variables[key] = []
            return self.variables[key]
        self.variables[key] = value
        if self.autosave:
            self.save()
        
