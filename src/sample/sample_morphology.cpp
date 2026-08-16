#include <iostream>
#include <inspirecv/inspirecv.h>
// #include <inspirecv/backends/okcv/io/stb_wrapper.h>

int main() {
    inspirecv::Image image = inspirecv::Image::Create();
    image.Read("../images/erode_before.jpg", 1);

    inspirecv::Image eroded = image.Erode(79, 1);
    eroded.Show("erode");

    eroded.Write("dilate.jpg");

    inspirecv::Image dilated = image.Dilate(79, 1);
    dilated.Show("dilate");

    dilated.Write("dilate.jpg");

    return 0;
}
