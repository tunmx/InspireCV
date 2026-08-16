#include <iostream>
#include <inspirecv/inspirecv.h>
// #include <inspirecv/backends/okcv/io/stb_wrapper.h>

int main() {
    inspirecv::ImageT<float> image = inspirecv::ImageT<float>::Create();
    image.Read("../images/kun.jpg");

    inspirecv::ImageT<float> blurred;
    inspirecv::TimeSpend ts(inspirecv::GetCVBackend());
    for (int i = 0; i < 100; ++i) {  
        ts.Start();
        blurred = image.GaussianBlur(11, 0.0);
        ts.Stop();
    }
    std::cout << ts << std::endl;
    blurred.Write("gaussian_blur.jpg");

    return 0;
}
