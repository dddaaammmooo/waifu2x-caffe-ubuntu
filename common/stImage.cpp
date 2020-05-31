#include "stImage.h"
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

const int YToRGBConvertMode = cv::COLOR_GRAY2RGB;
const int YToRGBConverInversetMode = cv::COLOR_RGB2GRAY;
const int BGRToYConvertMode = cv::COLOR_BGR2YUV;
const int BGRToConvertInverseMode = cv::COLOR_YUV2BGR;

// float�ȉ摜��uint8_t�ȉ摜�ɕϊ�����ۂ̎l�̌ܓ��Ɏg���l
// https://github.com/nagadomi/waifu2x/commit/797b45ae23665a1c5e3c481c018e48e6f0d0e383
const double clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5);
const double clip_eps16 = (1.0 / 65535.0) * 0.5 - (1.0e-7 * (1.0 / 65535.0) * 0.5);
const double clip_eps32 = 1.0 * 0.5 - (1.0e-7 * 0.5);

const std::vector<stImage::stOutputExtentionElement> stImage::OutputExtentionList =
{
	{L".png",{8, 16}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".bmp",{8}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".jpg",{8}, 0, 100, 95, cv::IMWRITE_JPEG_QUALITY},
	{L".jp2",{8, 16}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".sr",{8}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".tif",{8, 16, 32}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".hdr",{8, 16, 32}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".exr",{8, 16, 32}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".ppm",{8, 16}, boost::optional<int>(), boost::optional<int>(), boost::optional<int>(), boost::optional<int>()},
	{L".webp",{8}, 1, 100, 100, cv::IMWRITE_WEBP_QUALITY},
	{L".tga",{8}, 0, 1, 0, 0},
};


template<typename BufType>
static bool readFile(boost::iostreams::stream<boost::iostreams::file_descriptor_source> &is, std::vector<BufType> &buf)
{
	if (!is)
		return false;

	const auto size = is.seekg(0, std::ios::end).tellg();
	is.seekg(0, std::ios::beg);

	buf.resize((size / sizeof(BufType)) + (size % sizeof(BufType)));
	is.read(buf.data(), size);
	if (is.gcount() != size)
		return false;

	return true;
}

template<typename BufType>
static bool readFile(const boost::filesystem::path &path, std::vector<BufType> &buf)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor_source> is;

	try
	{
		is.open(path, std::ios_base::in | std::ios_base::binary);
	}
	catch (...)
	{
		return false;
	}

	return readFile(is, buf);
}

template<typename BufType>
static bool writeFile(boost::iostreams::stream<boost::iostreams::file_descriptor> &os, const std::vector<BufType> &buf)
{
	if (!os)
		return false;

	const auto WriteSize = sizeof(BufType) * buf.size();
	os.write((const char *)buf.data(), WriteSize);
	if (os.fail())
		return false;

	return true;
}

template<typename BufType>
static bool writeFile(const boost::filesystem::path &path, std::vector<BufType> &buf)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor> os;

	try
	{
		os.open(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
	}
	catch (...)
	{
		return false;
	}

	return writeFile(os, buf);
}

static void Waifu2x_stbi_write_func(void *context, void *data, int size)
{
	boost::iostreams::stream<boost::iostreams::file_descriptor> *osp = (boost::iostreams::stream<boost::iostreams::file_descriptor> *)context;
	osp->write((const char *)data, size);
}

int stImage::DepthBitToCVDepth(const int depth_bit)
{
	switch (depth_bit)
	{
	case 8:
		return CV_8U;

	case 16:
		return CV_16U;

	case 32:
		return CV_32F;
	}

	// �s�������ǂƂ肠����CV_8U��Ԃ��Ă���
	return CV_8U;
}

double stImage::GetValumeMaxFromCVDepth(const int cv_depth)
{
	switch (cv_depth)
	{
	case CV_8U:
		return 255.0;

	case CV_16U:
		return 65535.0;

	case CV_32F:
		return 1.0;
	}

	// �s�������ǂƂ肠����255.0��Ԃ��Ă���
	return 255.0;
}

double stImage::GetEPS(const int cv_depth)
{
	switch (cv_depth)
	{
	case CV_8U:
		return clip_eps8;

	case CV_16U:
		return clip_eps16;

	case CV_32F:
		return clip_eps32;
	}

	// �s�������ǂƂ肠����clip_eps8�Ԃ��Ă���
	return clip_eps8;
}


Waifu2x::eWaifu2xError stImage::AlphaMakeBorder(std::vector<cv::Mat> &planes, const cv::Mat &alpha, const int offset)
{
	// ���̃J�[�l���Ɖ摜�̏􍞂݂��s���ƁA(x, y)�𒆐S�Ƃ���3�~3�̈�̍��v�l�����܂�
	const static cv::Mat sum2d_kernel = (cv::Mat_<double>(3, 3) <<
		1., 1., 1.,
		1., 1., 1.,
		1., 1., 1.);

	cv::Mat mask;
	cv::threshold(alpha, mask, 0.0, 1.0, cv::THRESH_BINARY); // �A���t�@�`�����l�����l�����ă}�X�N�Ƃ��Ĉ���

	cv::Mat mask_nega;
	cv::threshold(mask, mask_nega, 0.0, 1.0, cv::THRESH_BINARY_INV); // ���]�����}�X�N�i�l��1�̉ӏ��͊��S�����łȂ��L���ȉ�f�ƂȂ�j

	for (auto &p : planes) // ���S�ɓ����ȃs�N�Z���ɂ���S�~�����
	{
		p = p.mul(mask);
	}

	for (int i = 0; i < offset; i++)
	{
		cv::Mat mask_weight;
		cv::filter2D(mask, mask_weight, -1, sum2d_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT); // �}�X�N��3�~3�̈�̍��v�l�����߂�

		cv::Mat mask_nega_u8;
		mask_nega.convertTo(mask_nega_u8, CV_8U, 255.0, clip_eps8); // mask_nega��CV_U8�ŁiOpenCV��API��K�v�ɂȂ�j

		for (auto &p : planes) // 1�`�����l��������
		{
			// �`�����l����3�~3�̈���̗L����f�̕��ϒl�����߂�
			cv::Mat border;
			cv::filter2D(p, border, -1, sum2d_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
			border /= mask_weight;

			// �`�����l���̗L���ȉ�f�̕����ɁA�v�Z�������ϒl���R�s�[
			border.copyTo(p, mask_nega_u8);
		}

		// �}�X�N��1��c�����������̂�V�����}�X�N�Ƃ���(�}�X�N��3�~3�̈�̍��v�l�����߂����̂̔�0�̈�́A�}�X�N��1��c�����������̗̂̈�ɓ�����)
		cv::threshold(mask_weight, mask, 0.0, 1.0, cv::THRESH_BINARY);
		// �V�����}�X�N�̔��]�����}�X�N���v�Z
		cv::threshold(mask, mask_nega, 0.0, 1.0, cv::THRESH_BINARY_INV);
	}

	// ��f��0����1�ɃN���b�s���O
	for (auto &p : planes)
	{
		cv::threshold(p, p, 1.0, 1.0, cv::THRESH_TRUNC);
		cv::threshold(p, p, 0.0, 0.0, cv::THRESH_TOZERO);
	}

	return Waifu2x::eWaifu2xError_OK;
}

// �摜��ǂݍ���Œl��0.0f�`1.0f�͈̔͂ɕϊ�
Waifu2x::eWaifu2xError stImage::LoadMat(cv::Mat &im, const boost::filesystem::path &input_file)
{
	cv::Mat original_image;

	{
		std::vector<char> img_data;
		if (!readFile(input_file, img_data))
			return Waifu2x::eWaifu2xError_FailedOpenInputFile;

		const boost::filesystem::path ipext(input_file.extension());
		if (!boost::iequals(ipext.string(), ".bmp")) // ����̃t�@�C���`���̏ꍇOpenCV�œǂނƃo�O�邱�Ƃ�����̂�STBI��D�悳����
		{
			cv::Mat im(img_data.size(), 1, CV_8U, img_data.data());
			original_image = cv::imdecode(im, cv::IMREAD_UNCHANGED);

			if (original_image.empty())
			{
				const Waifu2x::eWaifu2xError ret = LoadMatBySTBI(original_image, img_data);
				if (ret != Waifu2x::eWaifu2xError_OK)
					return ret;
			}
		}
		else
		{
			const Waifu2x::eWaifu2xError ret = LoadMatBySTBI(original_image, img_data);
			if (ret != Waifu2x::eWaifu2xError_OK)
			{
				cv::Mat im(img_data.size(), 1, CV_8U, img_data.data());
				original_image = cv::imdecode(im, cv::IMREAD_UNCHANGED);
				if (original_image.empty())
					return ret;
			}
		}
	}

	im = original_image;

	return Waifu2x::eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError stImage::LoadMatBySTBI(cv::Mat &im, const std::vector<char> &img_data)
{
	int x, y, comp;
	stbi_uc *data = stbi_load_from_memory((const stbi_uc *)img_data.data(), img_data.size(), &x, &y, &comp, 0);
	if (!data)
		return Waifu2x::eWaifu2xError_FailedOpenInputFile;

	int type = 0;
	switch (comp)
	{
	case 1:
	case 3:
	case 4:
		type = CV_MAKETYPE(CV_8U, comp);
		break;

	default:
		return Waifu2x::eWaifu2xError_FailedOpenInputFile;
	}

	im = cv::Mat(cv::Size(x, y), type);

	const auto LinePixel = im.step1() / im.channels();
	const auto Channel = im.channels();
	const auto Width = im.size().width;
	const auto Height = im.size().height;

	assert(x == Width);
	assert(y == Height);
	assert(Channel == comp);

	auto ptr = im.data;
	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			for (int ch = 0; ch < Channel; ch++)
				ptr[(i * LinePixel + j) * comp + ch] = data[(i * x + j) * comp + ch];
		}
	}

	stbi_image_free(data);

	if (comp >= 3)
	{
		// RGB������BGR�ɕϊ�
		for (int i = 0; i < y; i++)
		{
			for (int j = 0; j < x; j++)
				std::swap(ptr[(i * LinePixel + j) * comp + 0], ptr[(i * LinePixel + j) * comp + 2]);
		}
	}

	return Waifu2x::eWaifu2xError_OK;
}

cv::Mat stImage::ConvertToFloat(const cv::Mat &im)
{
	cv::Mat convert;
	switch (im.depth())
	{
	case CV_8U:
		im.convertTo(convert, CV_32F, 1.0 / GetValumeMaxFromCVDepth(CV_8U));
		break;

	case CV_16U:
		im.convertTo(convert, CV_32F, 1.0 / GetValumeMaxFromCVDepth(CV_16U));
		break;

	case CV_32F:
		convert = im; // ������0.0�`1.0�̂͂��Ȃ̂ŕϊ��͕K�v�Ȃ�
		break;
	}

	return convert;
}


stImage::stImage() : mIsRequestDenoise(false), pad_w1(0), pad_h1(0), pad_w2(0), pad_h2(0)
{
}

stImage::~stImage()
{
}

void stImage::Clear()
{
	mOrgFloatImage.release();
	mTmpImageRGB.release();
	mTmpImageA.release();
	mTmpImageAOneColor.release();
	mEndImage.release();
}

Waifu2x::eWaifu2xError stImage::Load(const boost::filesystem::path &input_file)
{
	Clear();

	Waifu2x::eWaifu2xError ret;

	cv::Mat im;
	ret = LoadMat(im, input_file);
	if (ret != Waifu2x::eWaifu2xError_OK)
		return ret;

	mOrgFloatImage = im;
	mOrgChannel = im.channels();
	mOrgSize = im.size();

	const boost::filesystem::path ip(input_file);
	const boost::filesystem::path ipext(ip.extension());

	const bool isJpeg = boost::iequals(ipext.string(), ".jpg") || boost::iequals(ipext.string(), ".jpeg");

	mIsRequestDenoise = isJpeg;

	return Waifu2x::eWaifu2xError_OK;
}

Waifu2x::eWaifu2xError stImage::Load(const void* source, const int width, const int height, const int channel, const int stride)
{
	Clear();

	cv::Mat original_image(cv::Size(width, height), CV_MAKETYPE(CV_8U, channel), (void *)source, stride);

	if (original_image.channels() >= 3) // RGB�Ȃ̂�BGR�ɂ���
	{
		std::vector<cv::Mat> planes;
		cv::split(original_image, planes);

		std::swap(planes[0], planes[2]);

		cv::merge(planes, original_image);
	}

	mOrgFloatImage = original_image;
	mOrgChannel = original_image.channels();
	mOrgSize = original_image.size();

	mIsRequestDenoise = false;

	return Waifu2x::eWaifu2xError_OK;
}

Factor stImage::GetScaleFromWidth(const int width) const
{
	return Factor((double)width, (double)mOrgSize.width);
}

Factor stImage::GetScaleFromHeight(const int height) const
{
	return Factor((double)height, (double)mOrgSize.height);
}

bool stImage::RequestDenoise() const
{
	return mIsRequestDenoise;
}

void stImage::Preprocess(const int input_plane, const int net_offset)
{
	mOrgFloatImage = ConvertToFloat(mOrgFloatImage);

	ConvertToNetFormat(input_plane, net_offset);
}

bool stImage::IsOneColor(const cv::Mat & im)
{
	assert(im.channels() == 1);

	const size_t Line = im.step1();
	const size_t Width = im.size().width;
	const size_t Height = im.size().height;

	if (Width == 0 && Height == 0)
		return true;

	const float *ptr = (const float *)im.data;
	const float color = ptr[0];

	for (size_t i = 0; i < Height; i++)
	{
		for (size_t j = 0; j < Width; j++)
		{
			const size_t pos = Line * i + j;

			if (ptr[pos] != color)
				return false;
		}
	}

	return true;
}

void stImage::ConvertToNetFormat(const int input_plane, const int alpha_offset)
{
	if (input_plane == 1) // Y���f��
	{
		if (mOrgFloatImage.channels() == 1) // 1ch�����Ȃ̂ł��̂܂�
			mTmpImageRGB = mOrgFloatImage;
		else // BGR�Ȃ̂ŕϊ�
		{
			mTmpImageRGB = mOrgFloatImage;

			if (mTmpImageRGB.channels() == 4) // BGRA�Ȃ̂�A�������o��
			{
				std::vector<cv::Mat> planes;
				cv::split(mTmpImageRGB, planes);

				mTmpImageA = planes[3];
				planes.resize(3);

				AlphaMakeBorder(planes, mTmpImageA, alpha_offset); // �����ȃs�N�Z���ƕs�����ȃs�N�Z���̋��E�����̐F���L����

				// CreateBrightnessImage()��BGR����Y�ɕϊ�����̂œ���RGB�ɕς�����͂��Ȃ�
				cv::merge(planes, mTmpImageRGB);
			}

			CreateBrightnessImage(mTmpImageRGB, mTmpImageRGB);
		}
	}
	else // RGB���f��
	{
		if (mOrgFloatImage.channels() == 1) // 1ch�����Ȃ̂�RGB�ɕϊ�
		{
			cv::cvtColor(mOrgFloatImage, mTmpImageRGB, YToRGBConvertMode);
			mOrgFloatImage.release();
		}
		else // BGR����RGB�ɕϊ�(A����������A�����o��)
		{
			std::vector<cv::Mat> planes;
			cv::split(mOrgFloatImage, planes);
			mOrgFloatImage.release();

			if (planes.size() == 4) // BGRA�Ȃ̂�A�������o��
			{
				mTmpImageA = planes[3];
				planes.resize(3);

				if (!IsOneColor(mTmpImageA))
				{
					AlphaMakeBorder(planes, mTmpImageA, alpha_offset); // �����ȃs�N�Z���ƕs�����ȃs�N�Z���̋��E�����̐F���L����

					// ���g��p��RGB�ɕϊ�
					cv::cvtColor(mTmpImageA, mTmpImageA, cv::COLOR_GRAY2RGB);
				}
				else
				{
					mTmpImageAOneColor = mTmpImageA;
					mTmpImageA.release();
				}
			}

			// BGR����RGB�ɂ���
			std::swap(planes[0], planes[2]);

			cv::merge(planes, mTmpImageRGB);
		}

		mOrgFloatImage.release();
	}
}

// �摜����P�x�̉摜�����o��
Waifu2x::eWaifu2xError stImage::CreateBrightnessImage(const cv::Mat &float_image, cv::Mat &im)
{
	if (float_image.channels() > 1)
	{
		cv::Mat converted_color;
		cv::cvtColor(float_image, converted_color, BGRToYConvertMode);

		std::vector<cv::Mat> planes;
		cv::split(converted_color, planes);

		im = planes[0];
		planes.clear();
	}
	else
		im = float_image;

	return Waifu2x::eWaifu2xError_OK;
}

bool stImage::HasAlpha() const
{
	return !mTmpImageA.empty();
}

void stImage::GetScalePaddingedRGB(cv::Mat &im, cv::Size_<int> &size, const int net_offset, const int outer_padding,
	const int crop_w, const int crop_h, const int scale)
{
	GetScalePaddingedImage(mTmpImageRGB, im, size, net_offset, outer_padding, crop_w, crop_h, scale);
}

void stImage::SetReconstructedRGB(cv::Mat &im, const cv::Size_<int> &size, const int inner_scale)
{
	SetReconstructedImage(mTmpImageRGB, im, size, inner_scale);
}

void stImage::GetScalePaddingedA(cv::Mat &im, cv::Size_<int> &size, const int net_offset, const int outer_padding,
	const int crop_w, const int crop_h, const int scale)
{
	GetScalePaddingedImage(mTmpImageA, im, size, net_offset, outer_padding, crop_w, crop_h, scale);
}

void stImage::SetReconstructedA(cv::Mat &im, const cv::Size_<int> &size, const int inner_scale)
{
	SetReconstructedImage(mTmpImageA, im, size, inner_scale);
}

void stImage::GetScalePaddingedImage(cv::Mat &in, cv::Mat &out, cv::Size_<int> &size, const int net_offset, const int outer_padding,
	const int crop_w, const int crop_h, const int scale)
{
	cv::Mat ret;

	if (scale > 1)
	{
		cv::Size_<int> zoom_size = in.size();
		zoom_size.width *= scale;
		zoom_size.height *= scale;

		cv::resize(in, ret, zoom_size, 0.0, 0.0, cv::INTER_NEAREST);
	}
	else
		ret = in;

	in.release();

	size = ret.size();

	PaddingImage(ret, net_offset, outer_padding, crop_w, crop_h, ret);

	out = ret;
}

// ���͉摜��(Photoshop�ł���)�L�����o�X�T�C�Y��output_size�̔{���ɕύX
// �摜�͍���z�u�A�]����cv::BORDER_REPLICATE�Ŗ��߂�
void stImage::PaddingImage(const cv::Mat &input, const int net_offset, const int outer_padding,
	const int crop_w, const int crop_h, cv::Mat &output)
{
	const auto pad_w1 = net_offset + outer_padding;
	const auto pad_h1 = net_offset + outer_padding;
	const auto pad_w2 = (int)ceil((double)input.size().width / (double)crop_w) * crop_w - input.size().width + net_offset + outer_padding;
	const auto pad_h2 = (int)ceil((double)input.size().height / (double)crop_h) * crop_h - input.size().height + net_offset + outer_padding;

	cv::copyMakeBorder(input, output, pad_h1, pad_h2, pad_w1, pad_w2, cv::BORDER_REPLICATE);
}

// �g��A�p�f�B���O���ꂽ�摜��ݒ�
void stImage::SetReconstructedImage(cv::Mat &dst, cv::Mat &src, const cv::Size_<int> &size, const int inner_scale)
{
	const cv::Size_<int> s(size * inner_scale);

	// �u���b�N�T�C�Y�p�̃p�f�B���O����蕥��(outer_padding�͍č\�z�̉ߒ��Ŏ�菜����Ă���)
	dst = src(cv::Rect(0, 0, s.width, s.height));

	src.release();
}

void stImage::Postprocess(const int input_plane, const Factor scale, const int depth)
{
	DeconvertFromNetFormat(input_plane);
	ShrinkImage(scale);

	// �l��0�`1�ɃN���b�s���O
	cv::threshold(mEndImage, mEndImage, 1.0, 1.0, cv::THRESH_TRUNC);
	cv::threshold(mEndImage, mEndImage, 0.0, 0.0, cv::THRESH_TOZERO);

	mEndImage = DeconvertFromFloat(mEndImage, depth);

	AlphaCleanImage(mEndImage);
}

void stImage::Postprocess(const int input_plane, const int width, const int height, const int depth)
{
	DeconvertFromNetFormat(input_plane);
	ShrinkImage(width, height);

	// �l��0�`1�ɃN���b�s���O
	cv::threshold(mEndImage, mEndImage, 1.0, 1.0, cv::THRESH_TRUNC);
	cv::threshold(mEndImage, mEndImage, 0.0, 0.0, cv::THRESH_TOZERO);

	mEndImage = DeconvertFromFloat(mEndImage, depth);

	AlphaCleanImage(mEndImage);
}

void stImage::DeconvertFromNetFormat(const int input_plane)
{
	if (input_plane == 1) // Y���f��
	{
		if (mOrgChannel == 1) // ���Ƃ���1ch�����Ȃ̂ł��̂܂�
		{
			mEndImage = mTmpImageRGB;
			mTmpImageRGB.release();
			mOrgFloatImage.release();
		}
		else // ���Ƃ���BGR�Ȃ̂Ŋ����A���S���Y���Ŋg�債��UV�Ɋg�債��Y�����̂��Ė߂�
		{
			std::vector<cv::Mat> color_planes;
			CreateZoomColorImage(mOrgFloatImage, mTmpImageRGB.size(), color_planes);
			mOrgFloatImage.release();

			color_planes[0] = mTmpImageRGB;
			mTmpImageRGB.release();

			cv::Mat converted_image;
			cv::merge(color_planes, converted_image);
			color_planes.clear();

			cv::cvtColor(converted_image, mEndImage, BGRToConvertInverseMode);
			converted_image.release();

			if (!mTmpImageA.empty()) // A������̂ō���
			{
				std::vector<cv::Mat> planes;
				cv::split(mEndImage, planes);

				planes.push_back(mTmpImageA);
				mTmpImageA.release();

				cv::merge(planes, mEndImage);
			}
			else if (!mTmpImageAOneColor.empty()) // �P�F��A��߂�
			{
				std::vector<cv::Mat> planes;
				cv::split(mEndImage, planes);

				cv::Size_<int> zoom_size = planes[0].size();

				// �}�[�W��̃T�C�Y�ɍ��킹��
				cv::resize(mTmpImageAOneColor, mTmpImageAOneColor, zoom_size, 0.0, 0.0, cv::INTER_NEAREST);

				planes.push_back(mTmpImageAOneColor);
				mTmpImageAOneColor.release();

				cv::merge(planes, mEndImage);
			}
		}
	}
	else // RGB���f��
	{
		// �����̒n�_��mOrgFloatImage�͋�

		if (mOrgChannel == 1) // ���Ƃ���1ch�����Ȃ̂Ŗ߂�
		{
			cv::cvtColor(mTmpImageRGB, mEndImage, YToRGBConverInversetMode);
			mTmpImageRGB.release();
		}
		else // ���Ƃ���BGR�Ȃ̂�RGB����߂�(A����������A�����̂��Ė߂�)
		{
			std::vector<cv::Mat> planes;
			cv::split(mTmpImageRGB, planes);
			mTmpImageRGB.release();

			if (!mTmpImageA.empty()) // A������̂ō���
			{
				// RGB����1ch�ɖ߂�
				cv::cvtColor(mTmpImageA, mTmpImageA, cv::COLOR_RGB2GRAY);

				planes.push_back(mTmpImageA);
				mTmpImageA.release();
			}
			else if (!mTmpImageAOneColor.empty()) // �P�F��A��߂�
			{
				cv::Size_<int> zoom_size = planes[0].size();

				// �}�[�W��̃T�C�Y�ɍ��킹��
				cv::resize(mTmpImageAOneColor, mTmpImageAOneColor, zoom_size, 0.0, 0.0, cv::INTER_NEAREST);

				planes.push_back(mTmpImageAOneColor);
				mTmpImageAOneColor.release();
			}

			// RGB����BGR�ɂ���
			std::swap(planes[0], planes[2]);

			cv::merge(planes, mEndImage);
		}
	}
}

void stImage::ShrinkImage(const Factor scale)
{
	const auto Width = scale.MultiNumerator(mOrgSize.width);
	const auto Height = scale.MultiNumerator(mOrgSize.height);

	//const cv::Size_<int> ns(mOrgSize.width * scale, mOrgSize.height * scale);
	const cv::Size_<int> ns((int)Width.toDouble(), (int)Height.toDouble());
	if (mEndImage.size().width != ns.width || mEndImage.size().height != ns.height)
	{
		int argo = cv::INTER_CUBIC;
		if (scale.toDouble() < 0.5)
			argo = cv::INTER_AREA;

		cv::resize(mEndImage, mEndImage, ns, 0.0, 0.0, argo);
	}
}

void stImage::ShrinkImage(const int width, const int height)
{
	const cv::Size_<int> ns(width, height);
	if (mEndImage.size().width != ns.width || mEndImage.size().height != ns.height)
	{
		const auto scale_width = (float)mEndImage.size().width / (float)ns.width;
		const auto scale_height = (float)mEndImage.size().height / (float)ns.height;

		int argo = cv::INTER_CUBIC;
		if (scale_width < 0.5 || scale_height < 0.5)
			argo = cv::INTER_AREA;

		cv::resize(mEndImage, mEndImage, ns, 0.0, 0.0, argo);
	}
}

cv::Mat stImage::DeconvertFromFloat(const cv::Mat &im, const int depth)
{
	const int cv_depth = DepthBitToCVDepth(depth);
	const double max_val = GetValumeMaxFromCVDepth(cv_depth);
	const double eps = GetEPS(cv_depth);

	cv::Mat ret;
	if (depth == 32) // �o�͂�float�`���Ȃ�ϊ����Ȃ�
		ret = im;
	else
		im.convertTo(ret, cv_depth, max_val, eps);

	return ret;
}

namespace
{
	template<typename T>
	void AlphaZeroToZero(std::vector<cv::Mat> &planes)
	{
		cv::Mat alpha(planes[3]);

		const T *aptr = (const T *)alpha.data;

		T *ptr0 = (T *)planes[0].data;
		T *ptr1 = (T *)planes[1].data;
		T *ptr2 = (T *)planes[2].data;

		const size_t Line = alpha.step1();
		const size_t Width = alpha.size().width;
		const size_t Height = alpha.size().height;

		for (size_t i = 0; i < Height; i++)
		{
			for (size_t j = 0; j < Width; j++)
			{
				const size_t pos = Line * i + j;

				if (aptr[pos] == (T)0)
					ptr0[pos] = ptr1[pos] = ptr2[pos] = (T)0;
			}
		}
	}
}

void stImage::AlphaCleanImage(cv::Mat &im)
{
	// ���S�����̃s�N�Z���̐F������(�����̓s����A���S�����̃s�N�Z���ɂ��F��t��������)
	// ���f���ɂ���Ă͉摜�S��̊��S�����̏ꏊ�ɂ����������l�̃A���t�@���L���邱�Ƃ�����B������������߂�cv_depth�֕ϊ����Ă��炱�̏������s�����Ƃɂ���
	// (������cv_depth��32�̏ꍇ���ƈӖ��͖�����)
	// TODO: ���f��(�Ⴆ��Photo)�ɂ���Ă�0�����Ȃ��摜��ϊ����Ă�0.000114856390�Ƃ��ɂȂ�̂ŁA�K�؂Ȓl�̃N���b�s���O���s���H
	if (im.channels() > 3)
	{
		std::vector<cv::Mat> planes;
		cv::split(im, planes);
		im.release();

		const auto depth = planes[0].depth();
		switch (depth)
		{
		case CV_8U:
			AlphaZeroToZero<uint8_t>(planes);
			break;

		case CV_16U:
			AlphaZeroToZero<uint16_t>(planes);
			break;

		case CV_32F:
			AlphaZeroToZero<float>(planes);
			break;

		case CV_64F:
			AlphaZeroToZero<double>(planes);
			break;
		}

		cv::merge(planes, im);
	}
}


// ���͉摜��zoom_size�̑傫����cv::INTER_CUBIC�Ŋg�債�A�F���݂̂��c��
Waifu2x::eWaifu2xError stImage::CreateZoomColorImage(const cv::Mat &float_image, const cv::Size_<int> &zoom_size, std::vector<cv::Mat> &cubic_planes)
{
	cv::Mat zoom_cubic_image;
	cv::resize(float_image, zoom_cubic_image, zoom_size, 0.0, 0.0, cv::INTER_CUBIC);

	cv::Mat converted_cubic_image;
	cv::cvtColor(zoom_cubic_image, converted_cubic_image, BGRToYConvertMode);
	zoom_cubic_image.release();

	cv::split(converted_cubic_image, cubic_planes);
	converted_cubic_image.release();

	// ����Y�����͎g��Ȃ��̂ŉ��
	cubic_planes[0].release();

	return Waifu2x::eWaifu2xError_OK;
}

cv::Mat stImage::GetEndImage() const
{
	return mEndImage;
}

Waifu2x::eWaifu2xError stImage::Save(const boost::filesystem::path &output_file, const boost::optional<int> &output_quality)
{
	return WriteMat(mEndImage, output_file, output_quality);
}

Waifu2x::eWaifu2xError stImage::WriteMat(const cv::Mat &im, const boost::filesystem::path &output_file, const boost::optional<int> &output_quality)
{
	const boost::filesystem::path ip(output_file);
	const std::string ext = ip.extension().string();

	if (boost::iequals(ext, ".tga"))
	{
		unsigned char *data = im.data;

		std::vector<unsigned char> rgbimg;
		if (im.channels() >= 3 || im.step1() != im.size().width * im.channels()) // RGB�p�o�b�t�@�ɃR�s�[(���邢�̓p�f�B���O���Ƃ�)
		{
			const auto Line = im.step1();
			const auto Channel = im.channels();
			const auto Width = im.size().width;
			const auto Height = im.size().height;

			rgbimg.resize(Width * Height * Channel);

			const auto Stride = Width * Channel;
			for (int i = 0; i < Height; i++)
				memcpy(rgbimg.data() + Stride * i, im.data + Line * i, Stride);

			data = rgbimg.data();
		}

		if (im.channels() >= 3) // BGR��RGB�ɕ��ёւ�
		{
			const auto Line = im.step1();
			const auto Channel = im.channels();
			const auto Width = im.size().width;
			const auto Height = im.size().height;

			auto ptr = rgbimg.data();
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
					std::swap(ptr[(i * Width + j) * Channel + 0], ptr[(i * Width + j) * Channel + 2]);
			}
		}

		boost::iostreams::stream<boost::iostreams::file_descriptor> os;

		try
		{
			os.open(output_file, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
		}
		catch (...)
		{
			return Waifu2x::eWaifu2xError_FailedOpenOutputFile;
		}

		if (!os)
			return Waifu2x::eWaifu2xError_FailedOpenOutputFile;

		// RLE���k�̐ݒ�
		bool isSet = false;
		const auto &OutputExtentionList = stImage::OutputExtentionList;
		for (const auto &elm : OutputExtentionList)
		{
			if (elm.ext == L".tga")
			{
				if (elm.imageQualitySettingVolume && output_quality)
				{
					stbi_write_tga_with_rle = *output_quality;
					isSet = true;
				}

				break;
			}
		}

		// �ݒ肳��Ȃ������̂Ńf�t�H���g�ɂ���
		if (!isSet)
			stbi_write_tga_with_rle = 1;

		if (!stbi_write_tga_to_func(Waifu2x_stbi_write_func, &os, im.size().width, im.size().height, im.channels(), data))
			return Waifu2x::eWaifu2xError_FailedOpenOutputFile;

		return Waifu2x::eWaifu2xError_OK;
	}

	try
	{
		const boost::filesystem::path op(output_file);
		const boost::filesystem::path opext(op.extension());

		std::vector<int> params;

		const auto &OutputExtentionList = stImage::OutputExtentionList;
		for (const auto &elm : OutputExtentionList)
		{
			if (elm.ext == opext)
			{
				if (elm.imageQualitySettingVolume && output_quality)
				{
					params.push_back(*elm.imageQualitySettingVolume);
					params.push_back(*output_quality);
				}

				break;
			}
		}

		std::vector<uchar> buf;
		cv::imencode(ext, im, buf, params);

		if (writeFile(output_file, buf))
			return Waifu2x::eWaifu2xError_OK;

	}
	catch (...)
	{
	}

	return Waifu2x::eWaifu2xError_FailedOpenOutputFile;
}
