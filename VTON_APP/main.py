import torch
from Utillity import Dataloader,Model_loader,UNet1architecture,UNet2architecture
from Utillity.Batchplotter import plot_three_batches


#Creating the instances of the model architectures  
Undressing_model_archi=UNet1architecture.Architecutre(in_channels=3,out_channels=3)
Dressing_model_archi=UNet2architecture.Architecture(in_channels=6,out_channels=3)

#Selecting the GPU for better Computaiton
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Model paths 
undressing_model_path="D:\\pytorch_projects\\VTON_APP\\Trained_model\\Unet(1).pth"
dressing_model_path="D:\\pytorch_projects\\VTON_APP\\Trained_model\\Unet(2).pth"

#Test image directories path
person_image_path = "D:\\pytorch_projects\\VTON_APP\\test_img\\image"
cloth_image_path = "D:\\pytorch_projects\\VTON_APP\\test_img\\cloth"

#selected pairs of the images and  clothes 
index_pairs=[(7,8),(7,3),(7,10)]
num_imgaes=len(index_pairs)
#Instanciating Dataloaders 
dataset=Dataloader.Unet_Dataset(person_image_path,cloth_image_path,index_pairs=index_pairs,Height=224,Width=224)

#creating batches from the dataset
person_tensor_batch,cloth_tensor_batch=Dataloader.load_batch_from_dataset(dataset)

#moving all the tensors too the device 
person_tensor_batch=Dataloader.move_tensor_to_device(person_tensor_batch,device=device)
cloth_tensor_batch=Dataloader.move_tensor_to_device(cloth_tensor_batch,device=device)

#loading the models
undressing_model_loader=Model_loader.LoadModel(Undressing_model_archi,undressing_model_path,device=device)
undressing_model=undressing_model_loader.get_model()
undressing_model_ouputs=undressing_model(person_tensor_batch)
#print(undressing_model_ouputs.shape)

dressing_model_loader=Model_loader.LoadModel(Dressing_model_archi,dressing_model_path,device=device)
dressing_model=dressing_model_loader.get_model()
dressing_model_inputs=Dataloader.fuse_batched_tensors(undressing_model_ouputs,cloth_tensor_batch)
#print(dressing_model_inputs.shape)

dressing_model_outputs=dressing_model(dressing_model_inputs)
#print(dressing_model_outputs.shape)

plot_three_batches(person_tensor_batch,cloth_tensor_batch,dressing_model_outputs,titles=("Person images ","Cloth images","Generatd images"),num_images=num_imgaes)
print(f"All {len(person_tensor_batch)} Images are Plotted !! ")