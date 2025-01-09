#include "slice/slice.hpp"

int main()
{
    slice::SliceImage instance;

    cv::Mat image = cv::imread("wallhaven-l8vp7y.jpg");

    int slice_num_h = 3;
    int slice_num_v = 3;
    auto results = instance.slice(tensor::cvimg(image), slice_num_h, slice_num_v, 0.1);

    for (int i = 0; i < slice_num_h; i++)
    {
        for (int j = 0; j < slice_num_v; j++)
        {
            int index = i * slice_num_v + j;
            std::cout << results[index].x << " " << results[index].y << std::endl;
            std::cout << results[index].w << " " << results[index].h << std::endl;
            std::string image_name = std::to_string(j) + std::to_string(i) + ".jpg";
            cv::imwrite(image_name, results[index].image);
        }
        
    }
    return 0;
}