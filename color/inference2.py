from __future__ import print_function
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, distributed
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from utils import *
from vae_inference import *
from dataloader import *
import argparse
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import shutil


import torchvision.utils as vutils


parser = argparse.ArgumentParser(description='H&E Autoencoder')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--nodes',type=int,default=1,metavar='N',help='number of nodes to utilize for training')
parser.add_argument('--gpus',type=int,default=4,metavar='N',help='number of GPUs to utilize per node (default: 4)')
parser.add_argument('--workers',type=int,default=16,metavar='N',help='number of CPUs to use in the pytorch dataloader (default: 16)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--patches',type=int,default=200,metavar='N',help='number of patches to sample per H&E image (default: 200)')
parser.add_argument('--patch-size',type=int,default=512,metavar='N',help='size of the patch X*Y where x=patch_size and y=patch_size (default: 512)')
parser.add_argument('--svs-dir',default='/data/luberjm/data/small/svs',metavar='S',help='SVS file to sample from if not using pre-saved coords (default: /data/luberjm/data/small/svs)')
parser.add_argument('--custom-coords-file',default='/home/luberjm/pl/code/patch_coords.data',metavar='S',help='add this flag to use a non-default coords file (default: patch_coords.data)')
parser.add_argument('--train-size',default='100',metavar='N',help='size of the training set (default: 100)')
parser.add_argument('--test-size',default='10',metavar='N',help='size of the training set, must be an even number (default: 10)')
parser.add_argument('--accelerator',default='gpu', metavar='S',help='gpu accelerator to use, use ddp for running in parallel (default: gpu)')
parser.add_argument('--logging-name',default='autoencoder', metavar='S',help='name to log this run under in tensorboard (default: autoencoder)')
parser.add_argument('--resnet',default='resnet18',metavar='S')
parser.add_argument('--enc-dim',default='512',metavar='N')
parser.add_argument('--latent-dim',default='256',metavar='N')
parser.add_argument('--first-conv',dest='first_conv',action='store_true')
parser.add_argument('--maxpool1',dest='maxpool1',action='store_true')
parser.add_argument('--read-coords',dest='read_coords',action='store_true',help='add this flag to read in previously sampled patch coordinates that pass QC from the default file \'patch_coords.data\'')
parser.add_argument('--write-coords', dest='write_coords', action='store_true',help='add this flag to write out sampled coordinates that pass QC to the default file \'patch_coords.data\', which can be preloaded to speed up training')


args = parser.parse_args()


# kwargs = {'batch_size':args.batch_size,'pin_memory':True,'num_workers':args.workers}
kwargs = {'batch_size':128,'pin_memory':True,'num_workers':10}


if __name__ == '__main__':
    
    # initialize transforms to perform on data
    transformations = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(size=64),transforms.ToTensor(),transforms.Normalize(mean=[0.5937,0.5937,0.5937,0.5937], std=[0.0810,0.0810,0.0810,0.0810])])
    
    
    tb_logger = TensorBoardLogger('/home/plk6560/Desktop/uta_cancer_research/color/tb_logs', name=args.logging_name, log_graph=False)
    rddp = False
    if args.accelerator == "ddp":
        rddp = True
    
    # create trainer to evaluate model
    # print("gpus=", args.gpus)
    # changed the gpus argument to devices and added the accelerator argument
    # trainer = pl.Trainer(max_epochs=args.epochs, replace_sampler_ddp=rddp,accelerator=args.accelerator,devices=args.gpus,logger=tb_logger,num_nodes=args.nodes,auto_lr_find=False,benchmark=True,fast_dev_run=False) #flush_logs_every_n_steps=1
    # using this code for testing
    
    
    # trainer = pl.Trainer(max_epochs=args.epochs, replace_sampler_ddp=rddp,accelerator="gpu",devices=torch.cuda.device_count(),logger=tb_logger,num_nodes=args.nodes,auto_lr_find=False,benchmark=True,fast_dev_run=False) #flush_logs_every_n_steps=1
    
    
    trainer = pl.Trainer(max_epochs=args.epochs, replace_sampler_ddp=rddp,accelerator="gpu",devices=1,logger=tb_logger,num_nodes=args.nodes,auto_lr_find=False,benchmark=True,fast_dev_run=False) #flush_logs_every_n_steps=1

    
    print("latent dimension =", args.latent_dim)

    # load model
    # create custom VAE
    print("initializing model")
    trained_model_module = customVAE(enc_type=args.resnet,first_conv=args.first_conv,maxpool1=args.maxpool1,enc_out_dim=args.enc_dim,latent_dim=args.latent_dim)
    
    print("enc_type=", args.resnet)
    print("first_conv=", args.first_conv)
    print("maxpool1=",args.maxpool1)
    print("enc_out_dim=",args.enc_dim)
    print("latent_dim=",args.latent_dim)


    # load model
    print("loading model")
    # trained_model = trained_model_module.load_from_checkpoint(checkpoint_path="epoch=19-step=37040.ckpt") # this files location can be changed later
    trained_model = trained_model_module.load_from_checkpoint(checkpoint_path="/home/data/vae_best_models/batch128_latent128/epoch=99-step=7400.ckpt", latent_dim=args.latent_dim) # this files location can be changed later
    # copying old dataloader and removing arguments: arps.patches, args.write_coords, args.read_coords, args.custom_coords_file
    # create new class in the dataloader.py file
    # I will need to load all patches and files in this folder sequentially
    
    # the inference loader takes in dataset_dir, cancer type, patch size, num workers, and transforms
    

    # srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=80G --gres=gpu:1 --pty /bin/bash
    # cd Desktop/uta_cancer_search/color
    # conda activate vae
    # python3 inference2.py --svs-dir /home/data/gdc --patch-size 64 --latent-dim 32
    # python3 inference2.py --svs-dir /home/data/gdc --patch-size 64
    # python3 inference2.py --batch-size 128 --epochs 100 --gpus 1 --nodes 1 --workers 32 --accelerator gpu --logging-name overlap_experiment_batch128_latent16 --patches 18 --patch-size 64 --train-size 9396 --test-size 1044 --enc-dim 2048 --latent-dim 16 --resnet resnet50 --read-coords --custom-coords-file /home/mxn2498/projects/uta_cancer_search/custom_coords/overlap.data --svs-dir /home/data/gdc
    # python3 inference2.py --svs-dir /home/data/gdc --patch-size 64 --latent-dim 16
    # in the color directory with the vae environment
    # TODO: add argument in arg parser for cancer type, for now it will be hard coded
    # TODO: try changing batch size to 1 and see if that is imoortant, and if so, look more into it
    # TODO: set shuffle to false, the dataloader function is the one that shuffles, the test step of VAE_inference is the one that outputs images 
    


    
    


    img_dir = "/home/data/not-gdc/latent-dim-%s" % (args.latent_dim)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    

    

    
    #### start loop to go through images one by one ####
    
    all_image_filenames = pd.read_csv(join(args.svs_dir, 'metadata.csv'))
    all_image_filenames = list(all_image_filenames.apply(lambda x: join(args.svs_dir, x.id, x.filename), axis=1))
    all_image_filenames.sort()


    for im in all_image_filenames:
        
        # create folder for patches
        if os.path.exists(img_dir + "/patches"):
            print("removing patches folder and contents")
            shutil.rmtree(img_dir + "/patches")
        
        print("creating patches folder")
        os.makedirs(img_dir + "/patches")
        


        original = pyvips.Image.tiffload(im)
        inference_data = InferenceSvsDatasetFromImage(im,"Bronchus and lung", args.patch_size,10,transforms=transformations)
        # sampler = SequentialSampler(inference_data)
        # inference_loader = torch.utils.data.DataLoader(inference_data, **kwargs, shuffle = False, sampler = sampler)
        inference_loader = torch.utils.data.DataLoader(inference_data, **kwargs, shuffle = False)
        #print("outside of DataLoader")
        
        fun = trainer.test(trained_model,inference_loader) # inference_loader used to be test_loader



        # combine generated patches
        file_list = []

        # for every image in patches directory, put them in a list and combine them
        for image_filename in os.listdir(img_dir + "/patches"):
            file_list.append(img_dir + "/patches/" + image_filename)
        file_list.sort()

        # pprint(file_list)

        img_list = []
        for img_patch_name in file_list:
            patch = pyvips.Image.tiffload(img_patch_name)
            img_list.append(patch)


        # create folder for rows
        if os.path.exists(img_dir + "/rows"):
            print("removing rows folder and contents")
            shutil.rmtree(img_dir + "/rows")
        
        print("creating rows folder")
        os.makedirs(img_dir + "/rows")



        # # combine images for each row and save image in rows directory
        horizontal_patches = math.floor(original.width/args.patch_size/128) * 128 # the number of 64x64 patches that are in each row, thus cuts out patches that are on the end, is a multiple of 128
        horizontal_batches = int(horizontal_patches/128)                          # the number of groups of 128 64x64 patches
        print("horizontal_patches:", horizontal_patches)
        print("horizontal_batches:", horizontal_batches)
        

        number_of_batches_total = 100
        print("total number of batches:", horizontal_batches*math.floor(original.height/args.patch_size))
        if horizontal_batches*math.floor(original.height/args.patch_size) < 100:
            number_of_batches_total = horizontal_batches*math.floor(original.height/args.patch_size)

        # number_to_join = math.floor(100/horizontal_batches) * horizontal_batches, there were errors because the number of batches being joined was less than 100 for smaller images
        # that is why I created the variable abobve 
        number_to_join = math.floor(number_of_batches_total/horizontal_batches) * horizontal_batches
        print("number_to_join:", number_to_join)

        print("iterating", math.ceil(len(img_list)/number_to_join),  "times")

        total_height = 0

        for i in range(math.ceil(len(img_list)/number_to_join)):
            
            print(number_to_join)
            
            end_row = int(horizontal_patches/128)
            
            
            print("saving row", i*number_to_join, min((i+1)*number_to_join, len(img_list)))

            
            print("batches:", horizontal_batches)
            row_image = pyvips.Image.arrayjoin(img_list[i*number_to_join:min((i+1)*number_to_join, len(img_list))], across = horizontal_batches) # change 13 to horizontal columns
            #pprint(img_list)
            # row_image = pyvips.Image.arrayjoin(img_list[0:1], across = horizontal_batches) # change 13 to horizontal columns
            print("height, width:", row_image.height, row_image.width)
            total_height += row_image.height
            row_image.tiffsave(img_dir + "/rows/row%03d.tiff" % (i))
        print("the total height is", total_height)

        row_filename_list = []
        # get filenames of all rows
        for row_filename in os.listdir(img_dir + "/rows"):
            row_filename_list.append(img_dir + "/rows/" + row_filename)
        row_filename_list.sort()

        pprint(row_filename_list)

        row_list = []
        for row_name in row_filename_list:
            row = pyvips.Image.tiffload(row_name)
            row_list.append(row)
            print("appended row", row_name)

        print(len(row_list))

        final = pyvips.Image.arrayjoin(row_list, across = 1) # change so that original combines all rows together
        print("all images successfully joined. Dimensions (h, w):", final.height, final.width)
        final = final.crop(0, 0, final.width, total_height)
        print("final image cropped. Dimensions:", final.height, final.width)
        final.write_to_file(img_dir + "/" + os.path.splitext(os.path.basename(im))[0] + ".tiff")
        
        print("final image name:", img_dir + "/" + os.path.splitext(os.path.basename(im))[0] + ".tiff")
        print("there are", final.height/64, "rows")
        
        
        # import image
        #image = pyvips.Image.tiffload(img_dir + "/" + os.path.splitext(os.path.basename(im))[0] + ".tiff")


        # get image height and width
        #print("height: ", image.height)
        #print("width:  ", image.width)
        print("ran")






        # create original version of large image
        
        
        # original = original.crop(0, 0, original.width , 64*5) # comment this line out to get the entire image. it is cropped because the full image is not able to be loaded and viewed properly on the desktop
        # original.write_to_file(img_dir + "/original.png")

        # exit() # only go through one iteration, remove this exit when 





    
    
    
