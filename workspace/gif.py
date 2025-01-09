import imageio

with imageio.get_writer(uri='test.gif', mode='I', fps=1) as writer:
    for i in range(4):
        for j in range(4):
            image_name = f"{j}{i}.jpg"
            writer.append_data(imageio.imread(image_name))