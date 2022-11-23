# patch generation and whole image reconstruction code

This patching pipeline takes advantage of an existing variational autoencoder (VAE) model that was trained on H&E slide images in our lab. I was tasked with creating a prediction pipline that could convert a gigabyte size H&E slide image into small patches, convert each of those patches into a latent space using the encoder from the VAE mentioned above, and then use all of the latent spaces to create a regenerated version of the original image using the decoder from the same VAE.
