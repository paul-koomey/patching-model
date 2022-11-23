from __future__ import print_function
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import join, isfile
from utils import *
from multiprocessing import Pool
import torch
import pyvips
import random
import pickle
import pandas as pd
from pprint import pprint # added by Paul to view image file names and test
import math # used to calculate patch size/coords
from PIL import Image


# dataloader for inference
# this one is different because it will load images sequentially rather than randomly
class InferenceSvsDatasetFromImage(Dataset):    
    
    # will be used for testing:
    # dataset_dir : /home/data/gdc
    # cancer_type : breast
    # patch_size  : 64


    # I am taking out num_patches, write_coords, coords_file_name, read_coords custom_coords_file
    # originally: def __init__(self, dataset_dir, cancer_type, patch_size, num_patches, num_workers, write_coords, coords_file_name, read_coords, custom_coords_file, transforms=None):
    def __init__(self, image_path, cancer_type, patch_size, num_workers, transforms=None):
        
        
        super(InferenceSvsDatasetFromImage, self).__init__()
        self.transforms = transforms
        self.patchSize = patch_size
        # self.numPatches = num_patches # should be able to delete this

        # for all files in imageFilenames, get patches for every one in order


        self.patch_coords = []

        
        # include patching code to generate coordinates for all images
        # Edge pathces will be excluded if they are not the exact patch_size x patch_size

        # find a way to get 

        image = pyvips.Image.tiffload(image_path)
        height, width = image.height, image.width # y, x
        vertical_patches = math.floor(height/self.patchSize) #
        horizontal_patches = math.floor(width/self.patchSize/128) * 128 # subtract two to be safe, there were errors in cropping during testing
        
        print("filename:", image_path)
        print("height:", height)
        print("width:", width) # y, x
        #print(vertical_patches * self.patchSize + self.patchSize, horizontal_patches * self.patchSize + self.patchSize)
        print("vertical patches:", vertical_patches)
        print("horizontal patches:", horizontal_patches)

        #for i in range(horizontal_patches):
        #    self.patch_coords.append((img, (i*self.patchSize, 5120))) # y, x

        #print((img, (0, 0)))

        print("starting at index: ")

        for j in range(vertical_patches):
            for i in range(horizontal_patches): # determines number of patches horizontally
                # self.patch_coords.append((img, (i*self.patchSize + 17000,j*self.patchSize + 17000))) # y, x
                self.patch_coords.append((image_path, (i*self.patchSize,j*self.patchSize))) # y, x
        #print(self.patch_coords[-1])

        print("length:", len(self.patch_coords)) # y, x
        print("number of batches to predict:", 5*horizontal_patches/128)

        # self.patch_coords = self.patch_coords[self.patchSize*10:self.patchSize*10+1*horizontal_patches] # get 30 rows
        #########self.patch_coords = self.patch_coords[:3*horizontal_patches] # get 3 rows



        print("length:", len(self.patch_coords)) # y, x

        


    
    # the _load_file and _img_to_tensor functions are used in the __getitem__ function to load the individual patch
    def _load_file(self,file):
        image = pyvips.Image.new_from_file(str(file))
        return image

    # we will likely need to keep this function to convert each image properly
    def _img_to_tensor(self,img,x,y):
        t = img.crop(x,y,self.patchSize,self.patchSize)
        t_np = vips2numpy(t)
        #tt_np = transforms.ToTensor()(t_np)
        out_t = self.transforms(t_np)
        out_t = out_t[:3,:,:]
        return out_t
    
    # these functions are staying
    def __getitem__(self, index):
        # has a flag for read coordinates or write coordinates
        fname, coord_tuple = self.patch_coords[index] # get the data for patch we need given its index
        img = self._load_file(fname) # load the file
        out = self._img_to_tensor(img,coord_tuple[0],coord_tuple[1]) # get time image as a tensor so it can be returned

        # I need to generate patch_coords so that it has indices for all patches containing the following:
        # an image file name and a tuple containing x and y coordinates 
        # print("grabbing", coord_tuple)
        return out, out.size()
    
    def __len__(self):
        return len(self.patch_coords)



# dataloader for inference
# this one is different because it will load images sequentially rather than randomly
class InferenceSvsDatasetFromFolder(Dataset):    
    
    # will be used for testing:
    # dataset_dir : /home/data/gdc
    # cancer_type : breast
    # patch_size  : 64


    # I am taking out num_patches, write_coords, coords_file_name, read_coords custom_coords_file
    # originally: def __init__(self, dataset_dir, cancer_type, patch_size, num_patches, num_workers, write_coords, coords_file_name, read_coords, custom_coords_file, transforms=None):
    def __init__(self, dataset_dir, cancer_type, patch_size, num_workers, transforms=None):
        
        
        super(InferenceSvsDatasetFromFolder, self).__init__()
        meta = pd.read_csv(join(dataset_dir, 'metadata.csv'))
        self.imageFilenames = list(meta[meta['primary_site']==cancer_type].apply(lambda x: join(dataset_dir, x.id, x.filename), axis=1)) # only the images we want are given, others are filtered out
        self.dirLength = len(self.imageFilenames)
        self.transforms = transforms
        self.patchSize = patch_size
        # self.numPatches = num_patches # should be able to delete this

        # for all files in imageFilenames, get patches for every one in order


        self.patch_coords = []

        

        for img in self.imageFilenames[:1]: # I am stopping at 

            # include patching code to generate coordinates for all images
            # Edge pathces will be excluded if they are not the exact patch_size x patch_size

            # find a way to get 

            image = pyvips.Image.new_from_file(img)
            height, width = image.height, image.width # y, x
            vertical_patches = math.floor(height/self.patchSize) #
            horizontal_patches = math.floor(width/self.patchSize/128) * 128 # subtract two to be safe, there were errors in cropping during testing
            
            print("filename:", img)
            print("height:", height)
            print("width:", width) # y, x
            #print(vertical_patches * self.patchSize + self.patchSize, horizontal_patches * self.patchSize + self.patchSize)
            print("vertical patches:", vertical_patches)
            print("horizontal patches:", horizontal_patches)

            #for i in range(horizontal_patches):
            #    self.patch_coords.append((img, (i*self.patchSize, 5120))) # y, x

            #print((img, (0, 0)))

            print("starting at index: ")

            for j in range(vertical_patches):
                for i in range(horizontal_patches): # determines number of patches horizontally
                    # self.patch_coords.append((img, (i*self.patchSize + 17000,j*self.patchSize + 17000))) # y, x
                    self.patch_coords.append((img, (i*self.patchSize,j*self.patchSize))) # y, x
            #print(self.patch_coords[-1])

        print("length:", len(self.patch_coords)) # y, x
        print("number of batches to predict:", 1*horizontal_patches/128)

        self.patch_coords = self.patch_coords[89856:89856+1*horizontal_patches] # get 30 rows
        
        print("length:", len(self.patch_coords)) # y, x
        


    
    # the _load_file and _img_to_tensor functions are used in the __getitem__ function to load the individual patch
    def _load_file(self,file):
        image = pyvips.Image.new_from_file(str(file))
        return image

    # we will likely need to keep this function to convert each image properly
    def _img_to_tensor(self,img,x,y):
        t = img.crop(x,y,self.patchSize,self.patchSize)
        t_np = vips2numpy(t)
        #tt_np = transforms.ToTensor()(t_np)
        out_t = self.transforms(t_np)
        out_t = out_t[:3,:,:]
        return out_t
    
    # these functions are staying
    def __getitem__(self, index):
        # has a flag for read coordinates or write coordinates
        fname, coord_tuple = self.patch_coords[index] # get the data for patch we need given its index
        img = self._load_file(fname) # load the file
        out = self._img_to_tensor(img,coord_tuple[0],coord_tuple[1]) # get time image as a tensor so it can be returned

        # I need to generate patch_coords so that it has indices for all patches containing the following:
        # an image file name and a tuple containing x and y coordinates 
        # print("grabbing", coord_tuple)
        return out, out.size()
    
    def __len__(self):
        return len(self.patch_coords) - 2



# old function

class SvsDatasetFromFolder(Dataset):    
    def __init__(self, dataset_dir, cancer_type, patch_size, num_patches, num_workers, write_coords, coords_file_name, read_coords, custom_coords_file, transforms=None):
        super(SvsDatasetFromFolder, self).__init__()
        meta = pd.read_csv(join(dataset_dir, 'metadata.csv'))
        self.imageFilenames = list(meta[meta['primary_site']==cancer_type].apply(lambda x: join(dataset_dir, x.id, x.filename), axis=1))
        self.dirLength = len(self.imageFilenames)
        self.transforms = transforms
        self.patchSize = patch_size
        self.numPatches = num_patches
        if read_coords:
            with open(custom_coords_file,'rb') as filehandle:
                self.patch_coords = pickle.load(filehandle)
                filehandle.close()
        if not read_coords:
            pool = Pool(processes=num_workers)
            print("pool")
            pool_out = pool.map(self._fetch_coords,self.imageFilenames)
            self.patch_coords = [elem for sublist in pool_out for elem in sublist]
            random.shuffle(self.patch_coords)
        if write_coords:
            with open(join('/home/mxn2498/projects/uta_cancer_search/custom_coords/', coords_file_name),'wb') as filehandle:
                pickle.dump(self.patch_coords,filehandle)
                filehandle.close()
        if not read_coords:
            assert len(self.patch_coords) == self.dirLength * self.numPatches               
    def _fetch_coords(self,fname):
        print(fname,flush=True)
        img = self._load_file(fname)
        patches = self._patching(img)
        dirs = [fname] * len(patches)
        return list(zip(dirs,patches))
    def _load_file(self,file):
        image = pyvips.Image.new_from_file(str(file))
        return image
    def _get_intersection(self,a_x,a_y,b_x,b_y): #tensors are row major
        if abs(a_x - b_x) < self.patchSize and abs(a_y - b_y) < self.patchSize:
            return True
        else:
            return False
    def _get_intersections(self,x,y,coords):
        if len(coords) == 0:
            return True
        else:
            ml = set(map(lambda b: self._get_intersection(b[0],b[1],x,y), coords))
            if True in ml:
                return False
            else: 
                return True
    def _filter_whitespace(self,tensor_3d):
        r = np.mean(np.array(tensor_3d[0]))
        g = np.mean(np.array(tensor_3d[1]))
        b = np.mean(np.array(tensor_3d[2]))
        channel_avg = np.mean(np.array([r,g,b]))
        if channel_avg < .82:
            return True
        else:
            return False
    def _img_to_tensor(self,img,x,y):
        t = img.crop(x,y,self.patchSize,self.patchSize)
        t_np = vips2numpy(t)
        #tt_np = transforms.ToTensor()(t_np)
        out_t = self.transforms(t_np)
        out_t = out_t[:3,:,:]
        return out_t
    def _patching(self, img):
        count = 0
        coords = []
        while count < self.numPatches: #[4, x , y] -> many [4, 512, 512]
                rand_i = random.randint(0,img.width-self.patchSize)
                rand_j = random.randint(0,img.height-self.patchSize)
                temp = self._img_to_tensor(img,rand_i,rand_j)
                if self._filter_whitespace(temp):
                    if self._get_intersections(rand_j,rand_i,coords):
                        coords.append((rand_i,rand_j))
                        count+=1
        return coords
    def __getitem__(self, index):
        fname, coord_tuple = self.patch_coords[index]
        img = self._load_file(fname)
        out = self._img_to_tensor(img,coord_tuple[0],coord_tuple[1])
        return out, out.size()
    def __len__(self):
        return len(self.patch_coords)

