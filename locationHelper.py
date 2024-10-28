import pycountry
import pycountry_convert as pc
import numpy as np

class LocationHelper:
    '''Helper class for location related functions
        Init with a list of regions and get the country names, country codes, and continent names for the same regions
    '''
    
    region_list = []
    pycountry_obj_dict = {}

    def __init__(self, region_list: list):
        self.region_list = region_list
        self.region_list = self.polish_country_names()
        self.fill_pycountry_objects()

    def get_country_names(self):
        '''Get the country code from the country name'''
        return [self.pycountry_obj_dict[c].name if not self.pycountry_obj_dict[c] is None else None for c in self.region_list]
    
    def get_country_codes3(self):
        '''Get the 3 letter country code from the country name'''
        return [self.pycountry_obj_dict[c].alpha_3 if not self.pycountry_obj_dict[c] is None else None for c in self.region_list]
    
    def get_continent_names(self):
        '''Get the continent names from the country names'''
        return [self.country_to_continent(self.pycountry_obj_dict[c].alpha_2) if not self.pycountry_obj_dict[c] is None else None for c in self.region_list]

    def polish_country_names(self):
        '''Polish the country names to match the pycountry names'''
        polished_countries = []
        for c in self.region_list:
            if type(c) != str:
                polished_countries.append(None)
                continue
            c.replace("&", "and")
            c.strip()

            if 'Ivory' in c:
                polished_countries.append("cote d'ivoire")
            elif "Turkey" in c:
                polished_countries.append("türkiye")
            elif "United States" in c:
                polished_countries.append("united states")
            else:
                polished_countries.append(c)
        return polished_countries
    
    def fill_pycountry_objects(self):
        '''Get the pycountry objects for the regions'''
        for c in self.region_list:
            if c is None:
                self.pycountry_obj_dict[c] = None
                continue
            if not c in self.pycountry_obj_dict.keys():
                try:
                    self.pycountry_obj_dict[c] = pycountry.countries.search_fuzzy(c)[0]  
                except LookupError:
                    print(f'Country {c} could not be resolved with pycountry fuzzy search.\nTry to modify it in polish_country_names()')
                    self.pycountry_obj_dict[c] = None
    
    def country_to_continent(self, country_alpha2):
        '''Get the continent name, given the country name (could be made more efficient)'''
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name
    
    
        