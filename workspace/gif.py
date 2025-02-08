import imageio

with imageio.get_writer(uri='test.gif', mode='I', fps=1, loop=0) as writer:
    for i in range(5):
        for j in range(3):
            image_name = f"{i}{j}.jpg"
            writer.append_data(imageio.imread(image_name))
