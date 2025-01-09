#include "slice/slice.hpp"


// 随机颜色生成函数
cv::Scalar generateRandomColor() 
{
    int r = rand() % 256; // 生成0到255之间的随机数
    int g = rand() % 256;
    int b = rand() % 256;
    return cv::Scalar(b, g, r); // OpenCV使用BGR格式
}

// 模拟矩形填充透明度
void addTransparency(cv::Mat& image, const cv::Rect& rect, const cv::Scalar& color, double alpha) {
    cv::Mat overlay = image.clone(); // 克隆一份原图
    cv::rectangle(overlay, rect, color, -1); // 填充矩形区域
    cv::addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image); // 合成带透明度的矩形
}

int main()
{
    slice::SliceImage instance;

    cv::Mat image = cv::imread("wallhaven-l8vp7y.jpg");

    auto results = instance.slice(tensor::cvimg(image), 640, 320, 0.1, 0.1);

    int slice_num_h = slice::calculateNumCuts(1920, 640, 0.1);
    int slice_num_v = slice::calculateNumCuts(1080, 320, 0.1);

    for (int i = 0; i < slice_num_h; i++)
    {
        for (int j = 0; j < slice_num_v; j++)
        {
            int index = i * slice_num_v + j;
            std::cout << results[index].x << " " << results[index].y << std::endl;
            std::cout << results[index].w << " " << results[index].h << std::endl;
            int x = (int)(results[index].x);
            int y = (int)(results[index].y);
            int w = (int)(results[index].w);
            int h = (int)(results[index].h);

            // 生成一个随机颜色用于矩形框填充
            cv::Scalar randomColor = generateRandomColor();
            cv::Scalar borderColor = randomColor;

            // 生成矩形的坐标
            cv::Rect rect(x, y, w, h);

            // 给每个矩形添加透明填充
            addTransparency(image, rect, randomColor, 0.3); // alpha值设为0.3来模拟透明度

            // 在图像上绘制矩形框，增加更厚的边框
            int lineThickness = 6; // 增加边框厚度，使框更加突出
            cv::rectangle(image, rect, borderColor, lineThickness);

            // 在矩形内部添加文本标记，帮助区分
            std::string label = "Slice " + std::to_string(i) + "," + std::to_string(j);
            cv::putText(image, label, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, borderColor, 2);

            // 保存带框的图片
            std::string image_name = std::to_string(j) + std::to_string(i) + ".jpg";
            cv::imwrite(image_name, image);
        }
        
    }
    cv::imwrite("rect.jpg", image);
    return 0;
}