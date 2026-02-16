// Standart Stuff
#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <fstream>
#include <cstdlib>
#include <vector>

// Parallelization Stuff
#include <omp.h>


#if defined _WIN32
        #include <intrin.h>
        namespace MPI
        {
            void Init(int &, char **& ){}
            void Finalize( void ){}
            struct nan 
            {
                int Get_size( void ) const { return 1; }
                int Get_rank( void ) const { return 0; }
                void Send( const void * v1, int v2, double v3, int v4, int v5 ) const {}
                void Send( const void * v1, int v2,  int v3, int v4, int v5 ) const {}
                void Send( const void * v1, int v2,  float v3, int v4, int v5 ) const {}
                void Recv( void * v1, int v2, int v3, int v4, int v5 ) const {}
                void Barrier( void ) const {}
            };
            struct Datatype {};
            nan COMM_WORLD;
            double DOUBLE;
            int INT;
            float FLOAT;
            int ANY_TAG;
        }
        int setenv (const char *__name, const char *__value, int __replace) { return 0; }
#endif

#ifndef _WIN32
    #include <pthread.h>
    #include "mpi.h"
#endif

bool constexpr UseMPI = true;

#define PI 3.14159265358979323846

// Drawing Stuff
#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

const olc::Pixel MENU_TITLE(2, 189, 36);
const olc::Pixel MENU_ITEM(255, 255, 255);


struct LineSegment {
    float x1, y1, x2, y2;
};

// Paralelized sections
// Converte a coordenada da tela para a coordenada do fractal
double map(double x, double oldLow, double oldHigh, double newLow, double newHigh)
{
    double oldRange = (x - oldLow)/(oldHigh - oldLow);
    return oldRange * (newHigh - newLow) + newLow;
}

void DivideFractal(double** pParam, 
                   const olc::vi2d&pixel_tl, const olc::vi2d& pixel_br, 
                   const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
                   unsigned int nMaxIteration, unsigned int nNodesSize,
                   double exit_code = 0)
{
    if(nNodesSize == 1)
    {
        pParam[0][0] = pixel_tl.x;
        pParam[0][1] = pixel_tl.y;
        pParam[0][2] = pixel_br.x;
        pParam[0][3] = pixel_br.y;
        pParam[0][4] = frac_real.x;
        pParam[0][5] = frac_real.y;
        pParam[0][6] = frac_imag.x;
        pParam[0][7] = frac_imag.y;
        pParam[0][8] = nMaxIteration;
        pParam[0][9] = exit_code;
    }

    else
    {
        olc::vi2d new_pixel_tl = pixel_tl;
        olc::vi2d new_pixel_br = { pixel_br.x / (int)nNodesSize, pixel_br.y};

        double frac_real_part = std::abs(frac_real.y - frac_real.x) / nNodesSize;
        olc::vd2d new_frac_real = {frac_real.x, frac_real.x + frac_real_part};

        for(int i = 0; i < nNodesSize; ++i)
        {
            pParam[i][0] = new_pixel_tl.x;
            pParam[i][1] = new_pixel_tl.y;
            pParam[i][2] = new_pixel_br.x;
            pParam[i][3] = new_pixel_br.y;
            pParam[i][4] = new_frac_real.x;
            pParam[i][5] = new_frac_real.y;
            pParam[i][6] = frac_imag.x;
            pParam[i][7] = frac_imag.y;
            pParam[i][8] = nMaxIteration;
            pParam[i][9] = exit_code;

            new_pixel_tl.x = new_pixel_br.x;
            new_pixel_br.x = new_pixel_br.x + (pixel_br.x / nNodesSize);
            
            new_frac_real = {new_frac_real.y, new_frac_real.y + frac_real_part};
        }
    }
}

// L-System stuff
enum class LSystemType {
    Standard,
    TreeAndLeaves,
    Fern,
    Daisy
};

struct LSystemConfig {
    std::string axiom;
    std::map<char, std::string> rules;
    int iterations;
    float angle;
};

LSystemConfig GetLSystemConfig(LSystemType type) {
    switch (type) {
        case LSystemType::Standard:
            return {"X", {{'X', "F[+X][-X]FX"}, {'F', "FF"}}, 5, 20.0f};
        case LSystemType::TreeAndLeaves:
            return {"A", {{'A', "F[+A][-A]F[+A][-A]FA"}, {'F', "FF"}}, 4, 22.5f};
        case LSystemType::Fern:
            return {"X", {{'X', "F[+X]F[-X]+X"}, {'F', "FF"}}, 6, 20.0f};
        case LSystemType::Daisy:
            return {"F+F+F+F+F+F+F+F", {{'F', "F+F--F+F"}}, 3, 45.0f};
        default:
            return {"X", {{'X', "F[+X][-X]FX"}, {'F', "FF"}}, 5, 20.0f};
    }
}

// FRACTAIS SEQUENCIAIS
void CreateMandelbrotSequential(const olc::vi2d& pixel_tl, const olc::vi2d& pixel_br, 
                       const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
                       int* pFractalIterations, unsigned int nMaxIteration)
{
    int nScreenHeightSize = pixel_br.y;
    for (int x = pixel_tl.x; x < (pixel_br.x - 1); x++)
    {
        for (int y = pixel_tl.y; y < pixel_br.y; y++)
        {
            double a = map(x, pixel_tl.x, pixel_br.x, frac_real.x, frac_real.y);
            double b = map(y, pixel_tl.y, pixel_br.y, frac_imag.x, frac_imag.y);

            int n = 0;

            double ca = a;
            double cb = b;

            while (n < nMaxIteration && (a*a + b*b) < 4.0 )
            {
                // z1 = z0^2 + c
                // z2 = c^2 + c
                //      c^2 = a^2 - b^2 + 2abi

                // C^2
                double aa = a*a - b*b;
                double bb = 2 * a * b;

                // C^2 + C
                a = aa + ca;
                b = bb + cb;

                n++;
            }
            pFractalIterations[(x * nScreenHeightSize) + y] = n;
        }
    }
}


void CreateFlowerMandelbrotSequential(const olc::vi2d& pixel_tl, const olc::vi2d& pixel_br, 
                       const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
                       int* pFractalIterations, unsigned int nMaxIteration, int d)
{
    int nScreenHeightSize = pixel_br.y;
    for (int x = pixel_tl.x; x < (pixel_br.x - 1); x++)
    {
        for (int y = pixel_tl.y; y < pixel_br.y; y++)
        {
            double a = map(x, pixel_tl.x, pixel_br.x, frac_real.x, frac_real.y);
            double b = map(y, pixel_tl.y, pixel_br.y, frac_imag.x, frac_imag.y);

            int n = 0;

            double ca = a;
            double cb = b;

            while (n < nMaxIteration && (a * a + b * b) < 4.0)
            {
                // Convert z = a + bi to polar
                double r = sqrt(a * a + b * b);
                double theta = atan2(b, a);

                // Raise z to the power d: z^d = r^d * e^(i*d*theta)
                double r_d = pow(r, d);
                double new_a = r_d * cos(d * theta) + ca;
                double new_b = r_d * sin(d * theta) + cb;

                a = new_a;
                b = new_b;
                n++;
            }
            pFractalIterations[(x * nScreenHeightSize) + y] = n;
        }
    }  
}

void CreateJuliaFlowerSequential(const olc::vi2d& pixel_tl, const olc::vi2d& pixel_br, 
    const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
    int* pFractalIterations, unsigned int nMaxIteration,
    double c_real, double c_imag) 
{
    for (int x = pixel_tl.x; x < pixel_br.x; x++) {
        for (int y = pixel_tl.y; y < pixel_br.y; y++) {
            double a = map(x, pixel_tl.x, pixel_br.x, frac_real.x, frac_real.y); // z.real
            double b = map(y, pixel_tl.y, pixel_br.y, frac_imag.x, frac_imag.y); // z.imag
            int n = 0;

            while (n < nMaxIteration && (a*a + b*b) < 4.0) {
                // Compute z⁶
                double a2 = a*a, b2 = b*b;
                double a4 = a2*a2, b4 = b2*b2;
                double a6 = a4*a2 - 15*a4*b2 + 15*a2*b4 - b4*b2;
                double b6 = 6*a4*a*b - 20*a2*a*b*b2 + 6*a*b4*b;

                // z⁶ + c (fixed Julia parameter)
                a = a6 + c_real;
                b = b6 + c_imag;
                n++;
            }
            pFractalIterations[x * pixel_br.y + y] = n;
        }
    }
}

void CreateLSystemSequential(olc::PixelGameEngine* pge, const std::string& lsystem, 
                           float angle, float step, const olc::Pixel& color) {
    float x = pge->ScreenWidth() / 2.0f;
    float y = pge->ScreenHeight() * .75f;
    float current_angle = -90.0f; // Start pointing upwards
    
    std::vector<std::pair<float, float>> stack;
    std::vector<LineSegment> lines;
    
    for (char cmd : lsystem) {
        if (cmd == 'F') {
            float new_x = x + step * cos(current_angle * PI / 180.0f);
            float new_y = y + step * sin(current_angle * PI / 180.0f);
            lines.push_back({x, pge->ScreenHeight() - y, new_x, pge->ScreenHeight() - new_y}); // Flip Y coordinates
            x = new_x;
            y = new_y;
        } else if (cmd == '+') {
            current_angle += angle;
        } else if (cmd == '-') {
            current_angle -= angle;
        } else if (cmd == '[') {
            stack.push_back({x, y});
            stack.push_back({current_angle, 0}); // Using second element for angle
        } else if (cmd == ']') {
            if (!stack.empty()) {
                current_angle = stack.back().first;
                stack.pop_back();
                y = stack.back().second;
                x = stack.back().first;
                stack.pop_back();
            }
        }
    }
    
    // Draw all line segments
    for (const auto& segment : lines) {
        pge->DrawLine(segment.x1, pge->ScreenHeight() - segment.y1, 
                     segment.x2, pge->ScreenHeight() - segment.y2, color);
    }
}


// FRACTAIS PARALELOS
void CreateMandelbrotParallel(const olc::vi2d& pixel_tl, const olc::vi2d& pixel_br, 
                       const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
                       int* pFractalIterations, unsigned int nMaxIteration)
{
    
    int nScreenHeightSize = pixel_br.y;

    auto CHUNK = (pixel_br.x - pixel_tl.x) / 128;
    #pragma omp parallel for schedule(dynamic, CHUNK) num_threads(omp_get_num_procs()) 
    for (int x = pixel_tl.x; x < (pixel_br.x - 1); x++)
    {
        for (int y = pixel_tl.y; y < pixel_br.y; y++)
        {
            double a = map(x, pixel_tl.x, pixel_br.x, frac_real.x, frac_real.y);
            double b = map(y, pixel_tl.y, pixel_br.y, frac_imag.x, frac_imag.y);

            int n = 0;

            double ca = a;
            double cb = b;

            while (n < nMaxIteration && (a*a + b*b) < 4 )
            {
                // z1 = z0^2 + c
                // z2 = c^2 + c
                //      c^2 = a^2 - b^2 + 2abi

                // C^2
                double aa = a*a - b*b;
                double bb = 2 * a * b;

                // C^2 + C
                a = aa + ca;
                b = bb + cb;

                n++;
            }
            pFractalIterations[(x * nScreenHeightSize) + y] = n;
        }
    }
}



void CreateFlowerMandelbrotParallel(const olc::vi2d& pixel_tl, const olc::vi2d& pixel_br, 
                       const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
                       int* pFractalIterations, unsigned int nMaxIteration, int d)
{
    int nScreenHeightSize = pixel_br.y;
    int chunk_size = (pixel_br.x - pixel_tl.x) / (16 * omp_get_max_threads()); // Dynamic chunk size

    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int x = pixel_tl.x; x < pixel_br.x; x++) {
        for (int y = pixel_tl.y; y < pixel_br.y; y++) {
            double a = map(x, pixel_tl.x, pixel_br.x, frac_real.x, frac_real.y);
            double b = map(y, pixel_tl.y, pixel_br.y, frac_imag.x, frac_imag.y);
            double ca = a, cb = b;
            int n = 0;

            while (n < nMaxIteration && (a*a + b*b) < 4.0) {
                // Convert z = a + bi to polar
                double r = sqrt(a * a + b * b);
                double theta = atan2(b, a);

                // z = z^d + c
                double r_d = pow(r, d);
                a = r_d * cos(d * theta) + ca;
                b = r_d * sin(d * theta) + cb;

                n++;
            }
            pFractalIterations[x * nScreenHeightSize + y] = n;
        }
    }
}

void CreateJuliaFlowerParallel(const olc::vi2d& pixel_tl, const olc::vi2d& pixel_br, 
    const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
    int* pFractalIterations, unsigned int nMaxIteration,
    double c_real, double c_imag) 
{
    #pragma omp parallel for collapse(2)
    for (int x = pixel_tl.x; x < pixel_br.x; x++) {
        for (int y = pixel_tl.y; y < pixel_br.y; y++) {
            double a = map(x, pixel_tl.x, pixel_br.x, frac_real.x, frac_real.y); // z.real
            double b = map(y, pixel_tl.y, pixel_br.y, frac_imag.x, frac_imag.y); // z.imag
            int n = 0;

            while (n < nMaxIteration && (a*a + b*b) < 4.0) {
                // Compute z⁶
                double a2 = a*a, b2 = b*b;
                double a4 = a2*a2, b4 = b2*b2;
                double a6 = a4*a2 - 15*a4*b2 + 15*a2*b4 - b4*b2;
                double b6 = 6*a4*a*b - 20*a2*a*b*b2 + 6*a*b4*b;

                // z⁶ + c (fixed Julia parameter)
                a = a6 + c_real;
                b = b6 + c_imag;
                n++;
            }
            pFractalIterations[x * pixel_br.y + y] = n;
        }
    }
}

void CreateLSystemParallel(olc::PixelGameEngine* pge, const std::string& lsystem, 
                         float angle, float step, const olc::Pixel& color) {
    float x = pge->ScreenWidth() / 2.0f;
    float y = pge->ScreenHeight() * .75f;
    float current_angle = -90.0f; // Start pointing upwards
    
    std::vector<std::pair<float, float>> stack;
    std::vector<LineSegment> lines;
    
    for (char cmd : lsystem) {
        if (cmd == 'F') {
            float new_x = x + step * cos(current_angle * PI / 180.0f);
            float new_y = y + step * sin(current_angle * PI / 180.0f);
            lines.push_back({x, pge->ScreenHeight() - y, new_x, pge->ScreenHeight() - new_y}); // Flip Y coordinates
            x = new_x;
            y = new_y;
        } else if (cmd == '+') {
            current_angle += angle;
        } else if (cmd == '-') {
            current_angle -= angle;
        } else if (cmd == '[') {
            stack.push_back({x, y});
            stack.push_back({current_angle, 0}); // Using second element for angle
        } else if (cmd == ']') {
            if (!stack.empty()) {
                current_angle = stack.back().first;
                stack.pop_back();
                y = stack.back().second;
                x = stack.back().first;
                stack.pop_back();
            }
        }
    }
    
    // Draw all line segments
    #pragma omp parallel for
    for (const auto& segment : lines) {
        pge->DrawLine(segment.x1, pge->ScreenHeight() - segment.y1, 
                        segment.x2, pge->ScreenHeight() - segment.y2, color);
    }
}


// INSTRUÇÕES PARALELAS
void CreateMandelbrotParallelAVX(const olc::vi2d& pixel_tl, const olc::vi2d& pixel_br, 
                       const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
                       int* pFractalIterations, int nMaxIteration,
                       int nScreenHeightSize = 0)
{
    
    nScreenHeightSize = pixel_br.y;
    
    double x_scale = (frac_real.y - frac_real.x) / (double(pixel_br.x) - double(pixel_tl.x));
	double y_scale = (frac_imag.y - frac_imag.x) / (double(pixel_br.y) - double(pixel_tl.y));

    double x_pos = frac_real.x;

    // 64-bit "double" registers
    __m256d _aa, _bb, _ca, _cb, _a, _b, _zr2, _zi2, _two, _four, _mask1;

    // 64-bit "int" registers
    __m256i _n, _maxIt, _mask2, _c, _one;

    // start of Y
    __m256d _y_pos_offsets, _y_pos, _y_scale, _y_jump;

    _y_scale = _mm256_set1_pd(y_scale);
    _y_jump = _mm256_set1_pd(y_scale * 4);
    _y_pos_offsets = _mm256_set_pd(0, 1, 2, 3);
    _y_pos_offsets = _mm256_mul_pd(_y_pos_offsets, _y_scale);

    // | 32.0 | 32.0 | 32.0 | 32.0 | 
    _maxIt = _mm256_set1_epi64x(nMaxIteration);

    // | 1.0 | 1.0 | 1.0 | 1.0 | 
    _one = _mm256_set1_epi64x(1);

    // | 2.0 | 2.0 | 2.0 | 2.0 | 
    _two = _mm256_set1_pd(2.0);

    // | 4.0 | 4.0 | 4.0 | 4.0 | 
    _four = _mm256_set1_pd(4.0);

    auto numProcs = omp_get_num_procs();

    auto CHUNK = (pixel_br.x - pixel_tl.x) / 128;
    #pragma omp parallel for schedule(dynamic, CHUNK) num_threads(numProcs) private(_n, _y_pos, x_pos, _c, _ca, _cb, _a, _b, _aa, _bb, _zr2, _zi2, _mask1, _mask2)
    for (int x = pixel_tl.x; x < (pixel_br.x - 1); x++)
    {
        // Calc start x
        x_pos = (frac_real.x + ((x) * x_scale));

        // Reset y position
        _bb =  _mm256_set1_pd(frac_imag.x);
        _y_pos = _mm256_add_pd(_bb, _y_pos_offsets);

        _ca = _mm256_set1_pd(x_pos);

        for (int y = pixel_tl.y; y < pixel_br.y; y += 4)
        {

            _a = _mm256_setzero_pd();
            _b = _mm256_setzero_pd();

            _n = _mm256_setzero_si256();

            _cb = _y_pos;

            repeat:

            // double ca = a;
            // double cb = b;

            // double aa = a*a - b*b;
            // double bb = 2 * a * b;

            // a = aa + ca;
            // b = bb + cb;

            // Multiply 256-bit registers in parallel, as they are doubles

            // a * a
            _zr2 = _mm256_mul_pd(_a, _a); // a * a

            // b * b
            _zi2 = _mm256_mul_pd(_b, _b); // b * b

            // a*a - b*b
            _aa = _mm256_sub_pd(_zr2, _zi2); // (a * a) - (b * a)

            // a * b
            _bb = _mm256_mul_pd(_a, _b); // a * b

            // (bb * 2)
            _b = _mm256_mul_pd(_bb, _two); // ((a * b) * 2)

            // (bb) + cb
            _b = _mm256_add_pd(_b, _cb); // ((a * b) * 2) + cb

            // aa + ca
            _a = _mm256_add_pd(_aa, _ca); // ((a * a) - (b * b)) + ca


            // while ((zr2 + zi2) < 4.0 && n < nMaxIteration)

            // aa = (a * a + b * b)
            _aa = _mm256_add_pd(_zr2, _zi2);

            // m1 = if(aa < 4.0)
            _mask1 = _mm256_cmp_pd(_aa, _four, _CMP_LT_OQ);

            // m2 = (nMaxIteration > n)
            _mask2 = _mm256_cmpgt_epi64(_maxIt, _n);

            // m2 = m1 AND m2 = if(aa < 4.0 && nMaxIterations > n)
            _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));

            // mask2 AND 00...01
            // mask2 = |00...00|11...11|00...00|11...11|
            // one   = |00...01|00...01|00...01|00...01|
            // c     = |00...00|00...01|00...00|00...00| // just the 2 element has to be incremented
            _c = _mm256_and_si256(_mask2, _one);

            // n + c
            _n = _mm256_add_epi64(_n, _c);

            // if ((a * a + b * b) < 4.0 && n < nMaxIterations) goto repeat
            if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0)
                goto repeat;

            #if defined _WIN32
                pFractalIterations[(x * nScreenHeightSize) + y + 0] = int(_mm256_extract_epi64(_n, 3));
                pFractalIterations[(x * nScreenHeightSize) + y + 1] = int(_mm256_extract_epi64(_n, 1));
                pFractalIterations[(x * nScreenHeightSize) + y + 2] = int(_mm256_extract_epi64(_n, 2));
                pFractalIterations[(x * nScreenHeightSize) + y + 3] = int(_mm256_extract_epi64(_n, 0));
            #endif
            #ifndef _WIN32
                pFractalIterations[(x * nScreenHeightSize) + y + 0] = int(_n[3]);
                pFractalIterations[(x * nScreenHeightSize) + y + 1] = int(_n[1]);
                pFractalIterations[(x * nScreenHeightSize) + y + 2] = int(_n[2]);
                pFractalIterations[(x * nScreenHeightSize) + y + 3] = int(_n[0]);
            #endif
            _y_pos = _mm256_add_pd(_y_pos, _y_jump);
        }
        // x_pos += x_scale;
    }
}

void CreateFlowerMandelbrotParallelAVX(const olc::vi2d& pixel_tl, const olc::vi2d& pixel_br, 
                       const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
                       int* pFractalIterations, unsigned int nMaxIteration, int d)
{
    int nScreenHeightSize = pixel_br.y;

    double x_scale = (frac_real.y - frac_real.x) / (double(pixel_br.x) - double(pixel_tl.x));
    double y_scale = (frac_imag.y - frac_imag.x) / (double(pixel_br.y) - double(pixel_tl.y));

    double x_pos = frac_real.x;

    // Registradores AVX
    __m256d _a, _b, _ca, _cb, _zr2, _zi2, _four;
    __m256i _n, _maxIt, _mask2, _c, _one;

    // Configurações iniciais
    __m256d _y_pos_offsets, _y_pos, _y_scale, _y_jump;

    _y_scale = _mm256_set1_pd(y_scale);
    _y_jump = _mm256_set1_pd(y_scale * 4);
    _y_pos_offsets = _mm256_set_pd(0, 1, 2, 3);
    _y_pos_offsets = _mm256_mul_pd(_y_pos_offsets, _y_scale);

    _maxIt = _mm256_set1_epi64x(nMaxIteration);
    _one = _mm256_set1_epi64x(1);
    _four = _mm256_set1_pd(4.0);

    auto numProcs = omp_get_num_procs();
    auto CHUNK = (pixel_br.x - pixel_tl.x) / 128;

    #pragma omp parallel for schedule(dynamic, CHUNK) num_threads(numProcs) \
    private(_n, _y_pos, x_pos, _c, _ca, _cb, _a, _b, _zr2, _zi2, _mask2)
    for (int x = pixel_tl.x; x < (pixel_br.x - 1); x++)
    {
        // Calcular posição x
        x_pos = (frac_real.x + ((x) * x_scale));
        _y_pos = _mm256_add_pd(_mm256_set1_pd(frac_imag.x), _y_pos_offsets);
        _ca = _mm256_set1_pd(x_pos);

        for (int y = pixel_tl.y; y < pixel_br.y; y += 4)
        {
            _a = _mm256_setzero_pd();
            _b = _mm256_setzero_pd();
            _n = _mm256_setzero_si256();
            _cb = _y_pos;

        repeat:
            // Guarda z original (antes de atualizar)
            __m256d _orig_a = _a;
            __m256d _orig_b = _b;

            // Calcula z^d usando multiplicação complexa repetida (res = orig ^ d)
            __m256d res_a = _orig_a;
            __m256d res_b = _orig_b;

            for (int i = 1; i < d; ++i) {
                // tmp = res_a * orig_a - res_b * orig_b
                __m256d tmp1 = _mm256_mul_pd(res_a, _orig_a);
                __m256d tmp2 = _mm256_mul_pd(res_b, _orig_b);
                __m256d tmp = _mm256_sub_pd(tmp1, tmp2);

                // res_b_new = res_a * orig_b + res_b * orig_a
                __m256d tmp3 = _mm256_mul_pd(res_a, _orig_b);
                __m256d tmp4 = _mm256_mul_pd(res_b, _orig_a);
                __m256d res_b_new = _mm256_add_pd(tmp3, tmp4);

                res_a = tmp;
                res_b = res_b_new;
            }

            // Agora z^d + c
            _a = _mm256_add_pd(res_a, _ca);
            _b = _mm256_add_pd(res_b, _cb);

            // Verificar condição de escape: |z|^2 < 4.0
            _zr2 = _mm256_mul_pd(_a, _a);
            _zi2 = _mm256_mul_pd(_b, _b);
            __m256d _magnitude = _mm256_add_pd(_zr2, _zi2);

            // Máscaras e incremento
            __m256d _mask1 = _mm256_cmp_pd(_magnitude, _four, _CMP_LT_OQ);
            _mask2 = _mm256_castpd_si256(_mask1);
            _mask2 = _mm256_and_si256(_mask2, _mm256_cmpgt_epi64(_maxIt, _n));

            _c = _mm256_and_si256(_mask2, _one);
            _n = _mm256_add_epi64(_n, _c);

            // Continua enquanto alguma lane ainda ativa
            if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0)
                goto repeat;

            // Armazenar resultados (portável)
            long long tmp_n[4];
            _mm256_storeu_si256((__m256i*)tmp_n, _n);

            pFractalIterations[(x * nScreenHeightSize) + y + 0] = int(tmp_n[3]);
            pFractalIterations[(x * nScreenHeightSize) + y + 1] = int(tmp_n[1]);
            pFractalIterations[(x * nScreenHeightSize) + y + 2] = int(tmp_n[2]);
            pFractalIterations[(x * nScreenHeightSize) + y + 3] = int(tmp_n[0]);

            _y_pos = _mm256_add_pd(_y_pos, _y_jump);
        }
    }
}

void CreateJuliaFlowerParallelAVX(const olc::vi2d& pixel_tl, const olc::vi2d& pixel_br, 
                       const olc::vd2d& frac_real, const olc::vd2d& frac_imag,
                       int* pFractalIterations, unsigned int nMaxIteration,
                       double c_real, double c_imag)
{
    int nScreenHeightSize = pixel_br.y;
    
    double x_scale = (frac_real.y - frac_real.x) / (double(pixel_br.x) - double(pixel_tl.x));
    double y_scale = (frac_imag.y - frac_imag.x) / (double(pixel_br.y) - double(pixel_tl.y));

    double x_pos = frac_real.x;

    // Registradores AVX
    __m256d _a, _b, _a2, _b2, _a4, _b4, _a6, _b6;
    __m256d _temp1, _temp2, _temp3, _temp4;
    __m256d _zr2, _zi2, _four, _mask1;
    __m256d _c_real, _c_imag;
    __m256i _n, _maxIt, _mask2, _c, _one;

    // Configurações iniciais
    __m256d _y_pos_offsets, _y_pos, _y_scale, _y_jump;

    _y_scale = _mm256_set1_pd(y_scale);
    _y_jump = _mm256_set1_pd(y_scale * 4);
    _y_pos_offsets = _mm256_set_pd(0, 1, 2, 3);
    _y_pos_offsets = _mm256_mul_pd(_y_pos_offsets, _y_scale);

    _maxIt = _mm256_set1_epi64x(nMaxIteration);
    _one = _mm256_set1_epi64x(1);
    _four = _mm256_set1_pd(4.0);
    _c_real = _mm256_set1_pd(c_real);
    _c_imag = _mm256_set1_pd(c_imag);

    // Constantes para cálculo de z⁶
    __m256d _fifteen = _mm256_set1_pd(15.0);
    __m256d _six = _mm256_set1_pd(6.0);
    __m256d _twenty = _mm256_set1_pd(20.0);

    auto numProcs = omp_get_num_procs();
    auto CHUNK = (pixel_br.x - pixel_tl.x) / 128;
    
    #pragma omp parallel for schedule(dynamic, CHUNK) num_threads(numProcs) \
    private(_n, _y_pos, x_pos, _c, _a, _b, _a2, _b2, _a4, _b4, _a6, _b6, \
            _temp1, _temp2, _temp3, _temp4, _zr2, _zi2, _mask1, _mask2)
    for (int x = pixel_tl.x; x < (pixel_br.x - 1); x++)
    {
        x_pos = (frac_real.x + ((x) * x_scale));
        __m256d _bb = _mm256_set1_pd(frac_imag.x);
        _y_pos = _mm256_add_pd(_bb, _y_pos_offsets);
        __m256d _ca = _mm256_set1_pd(x_pos);

        for (int y = pixel_tl.y; y < pixel_br.y; y += 4)
        {
            _a = _mm256_setzero_pd();
            _b = _mm256_setzero_pd();
            _n = _mm256_setzero_si256();

            // Inicializar z com as coordenadas do pixel
            _a = _ca;
            _b = _y_pos;

            repeat:

            // Calcular z² = (a² - b²) + i(2ab)
            _a2 = _mm256_mul_pd(_a, _a); // a²
            _b2 = _mm256_mul_pd(_b, _b); // b²
            
            // Calcular z⁴ = (z²)²
            // Parte real: (a² - b²)² - (2ab)²
            // Parte imaginária: 2*(a² - b²)*(2ab)
            __m256d _real_z2 = _mm256_sub_pd(_a2, _b2);
            __m256d _imag_z2 = _mm256_mul_pd(_mm256_mul_pd(_a, _b), _mm256_set1_pd(2.0));
            
            _a4 = _mm256_sub_pd(_mm256_mul_pd(_real_z2, _real_z2), 
                               _mm256_mul_pd(_imag_z2, _imag_z2));
            _b4 = _mm256_mul_pd(_mm256_mul_pd(_real_z2, _imag_z2), _mm256_set1_pd(2.0));

            // Calcular z⁶ = z⁴ * z²
            // a6 = a4*a2 - b4*b2
            // b6 = a4*b2 + b4*a2
            _a6 = _mm256_sub_pd(_mm256_mul_pd(_a4, _real_z2), 
                               _mm256_mul_pd(_b4, _imag_z2));
            _b6 = _mm256_add_pd(_mm256_mul_pd(_a4, _imag_z2), 
                               _mm256_mul_pd(_b4, _real_z2));

            // z⁶ + c
            _a = _mm256_add_pd(_a6, _c_real);
            _b = _mm256_add_pd(_b6, _c_imag);

            // Verificar condição de escape: |z| < 4.0
            _zr2 = _mm256_mul_pd(_a, _a);
            _zi2 = _mm256_mul_pd(_b, _b);
            __m256d _magnitude = _mm256_add_pd(_zr2, _zi2);
            
            _mask1 = _mm256_cmp_pd(_magnitude, _four, _CMP_LT_OQ);
            _mask2 = _mm256_cmpgt_epi64(_maxIt, _n);
            _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));
            
            _c = _mm256_and_si256(_mask2, _one);
            _n = _mm256_add_epi64(_n, _c);

            if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0)
                goto repeat;

            // Armazenar resultados
            #if defined _WIN32
                pFractalIterations[(x * nScreenHeightSize) + y + 0] = int(_mm256_extract_epi64(_n, 3));
                pFractalIterations[(x * nScreenHeightSize) + y + 1] = int(_mm256_extract_epi64(_n, 1));
                pFractalIterations[(x * nScreenHeightSize) + y + 2] = int(_mm256_extract_epi64(_n, 2));
                pFractalIterations[(x * nScreenHeightSize) + y + 3] = int(_mm256_extract_epi64(_n, 0));
            #else
                pFractalIterations[(x * nScreenHeightSize) + y + 0] = int(_n[3]);
                pFractalIterations[(x * nScreenHeightSize) + y + 1] = int(_n[1]);
                pFractalIterations[(x * nScreenHeightSize) + y + 2] = int(_n[2]);
                pFractalIterations[(x * nScreenHeightSize) + y + 3] = int(_n[0]);
            #endif
            
            _y_pos = _mm256_add_pd(_y_pos, _y_jump);
        }
    }
}


// L-System Functions
std::string GenerateLSystem(const std::string& axiom, const std::map<char, std::string>& rules, int iterations) {
    std::string current = axiom;
    for (int i = 0; i < iterations; i++) {
        std::string next;
        for (char c : current) {
            auto it = rules.find(c);
            if (it != rules.end()) {
                next += it->second;
            } else {
                next += c;
            }
        }
        current = next;
    }
    return current;
}

// CLASSE PRA DESENHO DO FRACTAL
class MandelbrotFractal : public olc::PixelGameEngine
{
public:
	MandelbrotFractal()
	{
		sAppName = "Fractais!!";
	}

protected:
    double julia_c_real = -0.7;  // Example: 6-petal Julia parameter
    double julia_c_imag = 0.27;
    int julia_preset = 1;

    int nWidth;
    int nHeight;

    int sideBarWidth = 400;

    int nMaxIteration = 32;
    int nColorMode = 0;
    int nFracMode = 0;
    int fracType = 0;
    int nPetals = 5;
    
    int lSys = 1;
    bool isTyping = false;
    std::string axiom = "";

    std::string sequence;

    int* pFractalIterations;

    // MPI Coord stuff
    int nMyRank, nNodesSize;

    double** pNodesParam;

    // L-System variables

    enum class InputStage { Axiom, Rules, Done };
    InputStage inputStage = InputStage::Done;

    std::string lsystemAxiom = "X";
    std::map<char, std::string> lsystemRules = {{'X', "F[+X][-X]FX"}, {'F', "FF"}};
    std::string currentRule;
    int lsystemIterations = 5;
    float lsystemAngle = 20.0f;
    /* 
    STANDARD
    std::string lsystemAxiom = "X";
    std::map<char, std::string> lsystemRules = {{'X', "F[+X][-X]FX"}, {'F', "FF"}};
    int lsystemIterations = 5;
    float lsystemAngle = 20.0f;

    TREE AND LEAVES
    axiom = "A"
    rules = {
        {'A', "F[+A][-A]F[+A][-A]FA"},
        {'F', "FF"}
    }
    angle = 22.5f
    iterations = 4
    
    FERN PATTERN
    axiom = "X"
    rules = {
        {'X', "F[+X]F[-X]+X"},
        {'F', "FF"}
    }
    angle = 20.0f
    iterations = 6

    DAISY
    axiom = "F+F+F+F+F+F+F+F"
    rules = {{'F', "F+F--F+F"}}
    angle = 45.0f
    iterations = 3
    */
    float lsystemBaseStep = 2.0f;  // This will be your reference size
    float lsystemStep = 1.0f;      // This will be the adjusted size
    std::string lsystemSequence;
    bool lsystemGenerated = false;

// UI variables
protected:

    bool toggleBenchmark = true;
    bool toggleHelp = true;
    bool toggleScreenShotView = false;
    bool toggleScreenShot = false;
    int nScreenShotCount = 0;

    std::string calcName;
    olc::Pixel color;
    olc::Pixel fracModeColor[3];
    olc::Pixel fracColorCol[3];
    olc::Pixel fracCol;

    olc::Pixel colorIterations = olc::WHITE;
    olc::Pixel zoomColor[2];
    olc::Pixel mouseColor;

public:
	bool OnUserCreate() override
	{
        char* ssc = getenv("SCREENSHOTCOUNT");
        if(ssc)
            nScreenShotCount = atoi(ssc);

		nWidth = ScreenWidth();
        nHeight = ScreenHeight();
        pFractalIterations = new int[ScreenWidth() * ScreenHeight()]{ 0 };

        // MPI
        if(!UseMPI)
        {
            nNodesSize = 1;
            nMyRank = 0;
        }
        else
        {
            nNodesSize = MPI::COMM_WORLD.Get_size();
            std::cout << "Nodes size: " << nNodesSize << "\n";
            nMyRank = MPI::COMM_WORLD.Get_rank();
        }

        pNodesParam = new double*[nNodesSize];
        for(int i = 0; i < nNodesSize; ++i)
            pNodesParam[i] = new double[10]{-1};


        // Initialize L-System
        lsystemSequence = GenerateLSystem(lsystemAxiom, lsystemRules, lsystemIterations);
        lsystemGenerated = true;
        
		return true;
	}

    
	bool OnUserUpdate(float fElapsedTime) override
	{
        //std::cout << "Rodando OnUserUpdate" << std::endl;  // Se isso parar de imprimir, significa que o loop parou

		// Panning and Zoomig, credits to @OneLoneCoder who i'am inpired for
        olc::vd2d vMouse = {(double)GetMouseX(), (double)GetMouseY()};

        // Get the position of the mouse and move the world Final Pos - Inital Pos
        // This make us drag Around the Screen Space, with the OffSet variable
        if(GetMouse(0).bPressed)
        {
            vStartPan = vMouse;
        }

        mouseColor = olc::WHITE;
        if(GetMouse(0).bHeld)
        {
            vOffset -= (vMouse - vStartPan) / vScale;
            vStartPan = vMouse;
            mouseColor = olc::RED;
        }

        olc::vd2d vMouseBeforeZoom;
        ScreenToWorld(vMouse, vMouseBeforeZoom);

        zoomColor[0] = olc::WHITE;
        zoomColor[1] = olc::WHITE;
        if (GetKey(olc::Key::E).bHeld) { vScale *= 1.1; zoomColor[0] = olc::RED; }
		if (GetKey(olc::Key::Q).bHeld) { vScale *= 0.9; zoomColor[1] = olc::RED; }
		
		olc::vd2d vMouseAfterZoom;
		ScreenToWorld(vMouse, vMouseAfterZoom);
		vOffset += (vMouseBeforeZoom - vMouseAfterZoom);

        // Now we have a smaller screen, and want to map to the world coord
        olc::vi2d pixel_tl = {  0,  0 };
        olc::vi2d pixel_br = {  nWidth,  nHeight };

        olc::vd2d frac_tl = { -2.0, -1.0 };
        olc::vd2d frac_br = {  1.0,  1.0 };

        // Then in the limits we now have the cartesian coords we want to draw
        // cartesian plane starting at top-left to bottom-right

        olc::vd2d frac_real;
        olc::vd2d frac_imag;

        ScreenToFrac(pixel_tl, pixel_br, frac_tl, frac_br, frac_real, frac_imag);

        //Hotkeys
        //Toggles
        if(!isTyping) {
            if(GetKey(olc::Key::H).bPressed) { toggleHelp = !toggleHelp; }
            if(GetKey(olc::Key::I).bPressed) { 
                if (fracType == 2) {
                    std::cout << "Insira o valor real: " << std::endl;
                    std::cin >> julia_c_real;
                    std::cout << "Insira o valor imaginario: " << std::endl;
                    std::cin >> julia_c_imag;
                } else if (fracType == 3) {
                    isTyping = true;
                    inputStage = InputStage::Axiom;
                    lSys = 1;
                    axiom = "";
                    currentRule = "";
                }
            }
            if(GetKey(olc::Key::S).bPressed) { toggleBenchmark = !toggleBenchmark;  }
            //toggleScreenShotView = !toggleScreenShotView;
            if(GetKey(olc::Key::P).bPressed) { TakeScreenshot(); }
            

            // Calculation Option
            if (GetKey(olc::Key::K1).bPressed) { nFracMode = 0; }
            if (GetKey(olc::Key::K2).bPressed) { nFracMode = 1; }
            if (GetKey(olc::Key::K3).bPressed) { nFracMode = 2; }
            // Color Option
            if (GetKey(olc::Key::F1).bPressed) nColorMode = 0;
            if (GetKey(olc::Key::F2).bPressed) nColorMode = 1;
            if (GetKey(olc::Key::F3).bPressed) nColorMode = 2;
            if (GetKey(olc::Key::F4).bPressed) nColorMode = 3;
            if (GetKey(olc::Key::F5).bPressed) nColorMode = 4;
            // Change Fractal
            if (GetKey(olc::Key::Z).bPressed) { fracType = 0; ResetFractalView(); }
            if (GetKey(olc::Key::X).bPressed) { fracType = 1; ResetFractalView(); }
            if (GetKey(olc::Key::C).bPressed) { fracType = 2; ResetFractalView(); }
            if (GetKey(olc::Key::V).bPressed) { fracType = 3; lsystemGenerated = false; ResetFractalView(); }

            if (fracType != 3) {
                // Modify the max iteration on the fly
                if (GetKey(olc::Key::EQUALS).bPressed || GetKey(olc::Key::NP_ADD).bPressed) { nMaxIteration += 32; colorIterations.g -= 16; colorIterations.b -= 16; }
                if (GetKey(olc::Key::MINUS).bPressed || GetKey(olc::Key::NP_SUB).bPressed) { nMaxIteration -= 32; colorIterations.g += 16; colorIterations.b += 16; }
                if (nMaxIteration < 32) { nMaxIteration = 32; colorIterations.g = 255; colorIterations.b = 255; }
                if (fracType == 1) {
                    if (GetKey(olc::Key::UP).bPressed)   { nPetals++; }
                    if (GetKey(olc::Key::DOWN).bPressed) { nPetals--; }
                    if (nPetals < 5) { nPetals = 5; }
                    if (nPetals > 10) { nPetals = 10; }
                } else if (fracType == 2) {
                    if (GetKey(olc::Key::K).bPressed) { //preset
                        if (julia_preset == 14) {
                            julia_preset = 0;
                        } else {
                            julia_preset++;
                        }

                        switch (julia_preset)
                        {
                        case 1:
                            julia_c_real = -1.06; julia_c_imag = 0.07; nMaxIteration = 32; break;
                        case 2:
                            julia_c_real = -0.7; julia_c_imag = 0.27; nMaxIteration = 32; break;
                        case 3:
                            julia_c_real = -0.68; julia_c_imag = 0.3; nMaxIteration = 32; break;
                        case 4:
                            julia_c_real = -0.52; julia_c_imag = -0.45; nMaxIteration = 32; break;
                        case 5:
                            julia_c_real = -0.44; julia_c_imag = -0.67; nMaxIteration = 32; break;
                        case 6:
                            julia_c_real = -0.26; julia_c_imag = -1.03; nMaxIteration = 32; break;
                        case 7:
                            julia_c_real = -0.23; julia_c_imag = -0.98; nMaxIteration = 32; break;  
                        case 8:
                            julia_c_real = -0.16; julia_c_imag = -0.84; nMaxIteration = 32; break;
                        case 9:
                            julia_c_real = -0.05; julia_c_imag = -0.79; nMaxIteration = 32; break;
                        case 10:
                            julia_c_real = 0.72; julia_c_imag = 0.43; nMaxIteration = 32; break;
                        case 11:
                            julia_c_real = 0.74; julia_c_imag = -0.26; nMaxIteration = 32; break;
                        case 12:
                            julia_c_real = 0.81; julia_c_imag = -0.41; nMaxIteration = 32; break;
                        case 13:
                            julia_c_real = 0.89; julia_c_imag = 0.66; nMaxIteration = 32; break;
                        case 14:
                            julia_c_real = 0.92; julia_c_imag = -0.45; nMaxIteration = 32; break;
                        }
                    }
                    if (GetKey(olc::Key::LEFT).bHeld)  julia_c_real -= 0.01;
                    if (GetKey(olc::Key::RIGHT).bHeld) julia_c_real += 0.01;
                    if (GetKey(olc::Key::UP).bHeld)    julia_c_imag += 0.01;
                    if (GetKey(olc::Key::DOWN).bHeld)  julia_c_imag -= 0.01;
                    if (GetKey(olc::Key::R).bPressed) { julia_c_real = -0.7; julia_c_imag = 0.27; nMaxIteration = 32; }
                    
                    // Clamp values to avoid extreme ranges
                    julia_c_real = std::clamp(julia_c_real, -1.5, 1.5);
                    julia_c_imag = std::clamp(julia_c_imag, -1.5, 1.5);
                    std::cout << "Julia c: " << julia_c_real << " + " << julia_c_imag << "i" << std::endl;
                }
            } else {
                if (GetKey(olc::Key::EQUALS).bPressed || GetKey(olc::Key::NP_ADD).bPressed) {
                    lsystemIterations++;
                    lsystemStep = lsystemBaseStep / pow(1.5f, lsystemIterations-3); // Adjust this factor as needed
                    lsystemGenerated = false;
                }
                if ((GetKey(olc::Key::MINUS).bPressed || GetKey(olc::Key::NP_SUB).bPressed) && lsystemIterations > 0) {
                    lsystemIterations--;
                    lsystemStep = lsystemBaseStep / pow(1.5f, lsystemIterations-3);
                    lsystemGenerated = false;
                }
                if (GetKey(olc::Key::LEFT).bHeld) lsystemAngle -= 1.0f;
                if (GetKey(olc::Key::RIGHT).bHeld) lsystemAngle += 1.0f;
                if (GetKey(olc::Key::Q).bHeld) lsystemStep *= 0.99f;
                if (GetKey(olc::Key::E).bHeld) lsystemStep *= 1.01f;
                if (GetKey(olc::Key::R).bPressed) {
                    lsystemAngle = 20.0f;
                    lsystemIterations = 5;
                    lsystemStep = lsystemBaseStep / pow(1.5f, lsystemIterations-3);
                    lsystemGenerated = false;
                }
                if (GetKey(olc::Key::K).bPressed) {
                    LSystemType selectedType;
                    lSys += 1;
                    if (lSys > 4) {
                        lSys = 1;
                    }
                    switch (lSys) {
                        case 1: 
                            selectedType = LSystemType::Standard; break;
                        case 2:
                            selectedType = LSystemType::TreeAndLeaves; break;
                        case 3:
                            selectedType = LSystemType::Fern; break;
                        case 4:
                            selectedType = LSystemType::Daisy; break;
                    }

                    LSystemConfig config = GetLSystemConfig(selectedType);
                    lsystemAxiom = config.axiom;
                    lsystemRules = config.rules;
                    lsystemAngle = config.angle;
                    lsystemIterations = config.iterations;

                    lsystemGenerated = false;
                }
            }
        }
        // Divide Fractal
        DivideFractal(pNodesParam, pixel_tl, pixel_br, frac_real, frac_imag, nMaxIteration, nNodesSize);

        // Cont the time with chrono clock
        auto tStart = std::chrono::high_resolution_clock::now();

        if (fracType != 3) {
            for(int i = 0; i < nNodesSize - 1; i++) {
                MPI::COMM_WORLD.Send((void*)pNodesParam[i], 10, MPI::DOUBLE, i+1, 0);
            }
        }

        switch (fracType)
        {
            case 0:
                switch (nFracMode)
                {
                case 0:
                    CreateMandelbrotSequential({(int)pNodesParam[nNodesSize-1][0], (int)pNodesParam[nNodesSize-1][1]}, 
                            {(int)pNodesParam[nNodesSize-1][2], (int)pNodesParam[nNodesSize-1][3]}, 
                            {pNodesParam[nNodesSize-1][4], pNodesParam[nNodesSize-1][5]}, 
                            {pNodesParam[nNodesSize-1][6], pNodesParam[nNodesSize-1][7]}, 
                            pFractalIterations, (unsigned int)pNodesParam[nNodesSize-1][8]); break;
                case 1:
                    CreateMandelbrotParallel({(int)pNodesParam[nNodesSize-1][0], (int)pNodesParam[nNodesSize-1][1]}, 
                            {(int)pNodesParam[nNodesSize-1][2], (int)pNodesParam[nNodesSize-1][3]}, 
                            {pNodesParam[nNodesSize-1][4], pNodesParam[nNodesSize-1][5]}, 
                            {pNodesParam[nNodesSize-1][6], pNodesParam[nNodesSize-1][7]}, 
                            pFractalIterations, (unsigned int)pNodesParam[nNodesSize-1][8]); break;
                case 2:
                    CreateMandelbrotParallelAVX({(int)pNodesParam[nNodesSize-1][0], (int)pNodesParam[nNodesSize-1][1]}, 
                            {(int)pNodesParam[nNodesSize-1][2], (int)pNodesParam[nNodesSize-1][3]}, 
                            {pNodesParam[nNodesSize-1][4], pNodesParam[nNodesSize-1][5]}, 
                            {pNodesParam[nNodesSize-1][6], pNodesParam[nNodesSize-1][7]}, 
                            pFractalIterations, (unsigned int)pNodesParam[nNodesSize-1][8]); break;
                }
                break;
            case 1:
                switch (nFracMode)
                {
                    case 0:
                        CreateFlowerMandelbrotSequential({(int)pNodesParam[nNodesSize-1][0], (int)pNodesParam[nNodesSize-1][1]}, 
                                {(int)pNodesParam[nNodesSize-1][2], (int)pNodesParam[nNodesSize-1][3]}, 
                                {pNodesParam[nNodesSize-1][4], pNodesParam[nNodesSize-1][5]}, 
                                {pNodesParam[nNodesSize-1][6], pNodesParam[nNodesSize-1][7]}, 
                                pFractalIterations, (unsigned int)pNodesParam[nNodesSize-1][8], nPetals); break;
                    case 1:
                        CreateFlowerMandelbrotParallel({(int)pNodesParam[nNodesSize-1][0], (int)pNodesParam[nNodesSize-1][1]}, 
                                {(int)pNodesParam[nNodesSize-1][2], (int)pNodesParam[nNodesSize-1][3]}, 
                                {pNodesParam[nNodesSize-1][4], pNodesParam[nNodesSize-1][5]}, 
                                {pNodesParam[nNodesSize-1][6], pNodesParam[nNodesSize-1][7]}, 
                                pFractalIterations, (unsigned int)pNodesParam[nNodesSize-1][8], nPetals); break;
                    case 2:
                        CreateFlowerMandelbrotParallelAVX({(int)pNodesParam[nNodesSize-1][0], (int)pNodesParam[nNodesSize-1][1]}, 
                                {(int)pNodesParam[nNodesSize-1][2], (int)pNodesParam[nNodesSize-1][3]}, 
                                {pNodesParam[nNodesSize-1][4], pNodesParam[nNodesSize-1][5]}, 
                                {pNodesParam[nNodesSize-1][6], pNodesParam[nNodesSize-1][7]}, 
                                pFractalIterations, (unsigned int)pNodesParam[nNodesSize-1][8], nPetals); break;
                }
                break;                
            case 2:
                switch (nFracMode)
                {
                    case 0:
                        CreateJuliaFlowerSequential({(int)pNodesParam[nNodesSize-1][0], (int)pNodesParam[nNodesSize-1][1]}, 
                                {(int)pNodesParam[nNodesSize-1][2], (int)pNodesParam[nNodesSize-1][3]}, 
                                {pNodesParam[nNodesSize-1][4], pNodesParam[nNodesSize-1][5]}, 
                                {pNodesParam[nNodesSize-1][6], pNodesParam[nNodesSize-1][7]}, 
                                pFractalIterations, (unsigned int)pNodesParam[nNodesSize-1][8], julia_c_real, julia_c_imag); break;
                    case 1:
                        CreateJuliaFlowerParallel({(int)pNodesParam[nNodesSize-1][0], (int)pNodesParam[nNodesSize-1][1]}, 
                                {(int)pNodesParam[nNodesSize-1][2], (int)pNodesParam[nNodesSize-1][3]}, 
                                {pNodesParam[nNodesSize-1][4], pNodesParam[nNodesSize-1][5]}, 
                                {pNodesParam[nNodesSize-1][6], pNodesParam[nNodesSize-1][7]}, 
                                pFractalIterations, (unsigned int)pNodesParam[nNodesSize-1][8], julia_c_real, julia_c_imag); break;
                    case 2:
                        CreateJuliaFlowerParallelAVX({(int)pNodesParam[nNodesSize-1][0], (int)pNodesParam[nNodesSize-1][1]}, 
                                {(int)pNodesParam[nNodesSize-1][2], (int)pNodesParam[nNodesSize-1][3]}, 
                                {pNodesParam[nNodesSize-1][4], pNodesParam[nNodesSize-1][5]}, 
                                {pNodesParam[nNodesSize-1][6], pNodesParam[nNodesSize-1][7]}, 
                                pFractalIterations, (unsigned int)pNodesParam[nNodesSize-1][8], julia_c_real, julia_c_imag); break;
                }
                break;
            case 3: // L-System
            {
                Clear(olc::BLACK);

                // Etapas de input
                if (isTyping)
                {
                    // --- INPUT DO AXIOMA ---
                    if (inputStage == InputStage::Axiom)
                    {
                        // Letras
                        for (int k = (int)olc::Key::A; k <= (int)olc::Key::Z; ++k) {
                            if (GetKey((olc::Key)k).bPressed) {
                                char base = 'a' + (k - (int)olc::Key::A);
                                base = char(toupper(base));
                                axiom.push_back(base);
                            }
                        }

                        // Outros caracteres
                        if (GetKey(olc::Key::OEM_4).bPressed) axiom.push_back('[');
                        if (GetKey(olc::Key::OEM_6).bPressed) axiom.push_back(']');
                        if (GetKey(olc::Key::NP_ADD).bPressed) axiom.push_back('+');
                        if (GetKey(olc::Key::MINUS).bPressed|| GetKey(olc::Key::NP_SUB).bPressed) axiom.push_back('-');
                        if (GetKey(olc::Key::BACK).bPressed && !axiom.empty()) axiom.pop_back();

                        if (GetKey(olc::Key::ENTER).bPressed || GetKey(olc::Key::ENTER).bPressed)
                        {
                            lsystemAxiom = axiom;
                            axiom.clear();
                            inputStage = InputStage::Rules;
                            std::cout << "Axioma: " << lsystemAxiom << std::endl;
                        }

                        break;
                    }

                    // --- INPUT DAS REGRAS ---
                    if (inputStage == InputStage::Rules)
                    {
                        // Captura caracteres válidos
                        for (int k = (int)olc::Key::A; k <= (int)olc::Key::Z; ++k) {
                            if (GetKey((olc::Key)k).bPressed) {
                                char base = 'a' + (k - (int)olc::Key::A);
                                base = char(toupper(base));
                                currentRule.push_back(base);
                            }
                        }
                        if (GetKey(olc::Key::EQUALS).bPressed) currentRule.push_back('=');
                        if (GetKey(olc::Key::OEM_4).bPressed) currentRule.push_back('[');
                        if (GetKey(olc::Key::OEM_6).bPressed) currentRule.push_back(']');
                        if (GetKey(olc::Key::NP_ADD).bPressed) currentRule.push_back('+');
                        if (GetKey(olc::Key::MINUS).bPressed || GetKey(olc::Key::NP_SUB).bPressed) currentRule.push_back('-');
                        if (GetKey(olc::Key::BACK).bPressed && !currentRule.empty()) currentRule.pop_back();

                        // ENTER confirma uma regra
                        if (GetKey(olc::Key::ENTER).bPressed || GetKey(olc::Key::ENTER).bPressed)
                        {
                            size_t pos = currentRule.find('=');
                            if (pos != std::string::npos && pos > 0 && pos < currentRule.size() - 1)
                            {
                                char symbol = currentRule[0];
                                std::string replacement = currentRule.substr(pos + 1);
                                lsystemRules[symbol] = replacement;
                                std::cout << "Regra adicionada: " << symbol << " -> " << replacement << std::endl;
                            }
                            currentRule.clear();
                        }

                        // R termina entrada de regras
                        if (GetKey(olc::Key::R).bPressed)
                        {
                            inputStage = InputStage::Done;
                            isTyping = false;
                            lsystemGenerated = false;
                            std::cout << "Regras finalizadas!" << std::endl;
                        }

                        break;
                    }
                }
                else
                {
                    // --- Etapa final: desenhar o L-System ---
                    if (inputStage == InputStage::Done && !lsystemGenerated)
                    {
                        lsystemSequence = GenerateLSystem(lsystemAxiom, lsystemRules, lsystemIterations);
                        lsystemGenerated = true;
                        std::cout << "L-System Sequence: " << lsystemSequence << std::endl;
                    }

                    if (lsystemGenerated)
                    {
                        switch (nFracMode)
                        {
                            case 0: CreateLSystemSequential(this, lsystemSequence, lsystemAngle, lsystemStep, olc::RED); break;
                            case 1: CreateLSystemParallel(this, lsystemSequence, lsystemAngle, lsystemStep, olc::RED); break;
                            case 2: CreateLSystemParallel(this, lsystemSequence, lsystemAngle, lsystemStep, olc::RED); break;
                        }
                    }

                    break;
                }
            }

        }

        if (fracType != 3) {
            for(int i = 0; i < nNodesSize - 1; i++) {
                MPI::COMM_WORLD.Recv((void*)(pFractalIterations + ((int)pNodesParam[i][0] * (int)pNodesParam[i][3])), 
                                    ((ScreenWidth()*ScreenHeight()) / nNodesSize), MPI::INT, i + 1, MPI::ANY_TAG);
            }
        }

        auto tEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> fTime = tEnd - tStart;

        //  PAINT
        if (fracType != 3) {
            switch (nColorMode)
            {
                case 0: 
                    // Render result to screen
                    if(toggleScreenShot)
                    {
                        std::ofstream screenShot("screen_shot_" + std::to_string(nScreenShotCount++) +".ppm");
                        screenShot << "P3\n" << std::to_string(ScreenHeight()) + " " + std::to_string(ScreenWidth()) + "\n" << "256\n";
                        for (int x = 0; x < ScreenWidth() - sideBarWidth; x++)
                        {
                            for (int y = 0; y < ScreenHeight(); y++)
                            {
                                int n = pFractalIterations[x * ScreenHeight() + y];
                                // Coloring Algorithm - Picked from https://solarianprogrammer.com/2013/02/28/mandelbrot-set-cpp-11/
                                double t = (double)n/(double)nMaxIteration;
                                // Use smooth polynomials for r, g, b
                                int rr = (int)(9*(1-t)*t*t*t*255);
                                int rg = (int)(15*(1-t)*(1-t)*t*t*255);
                                int rb =  (int)(8.5*(1-t)*(1-t)*(1-t)*t*255);
                                screenShot << rr << " " << rg << " " << rb << std::endl;
                            }
                        }
                        screenShot.close(); 
                        toggleScreenShot = false;
                    }
                    else
                    {         
                        #pragma omp parallel for
                        for (int x = 0; x < ScreenWidth() - sideBarWidth; x++)
                        {
                            for (int y = 0; y < ScreenHeight(); y++)
                            {
                                int n = pFractalIterations[x * ScreenHeight() + y];
                                // Coloring Algorithm - Picked from https://solarianprogrammer.com/2013/02/28/mandelbrot-set-cpp-11/
                                double t = (double)n/(double)nMaxIteration;
                                // Use smooth polynomials for r, g, b
                                int rr = (int)(9*(1-t)*t*t*t*255);
                                int rg = (int)(15*(1-t)*(1-t)*t*t*255);
                                int rb =  (int)(8.5*(1-t)*(1-t)*(1-t)*t*255);
                                Draw(x, y, olc::Pixel(rr, rg, rb, 255));
                            }
                        }
                    }
                    break;

                case 1:
                    if(toggleScreenShot)
                    {
                        std::ofstream screenShot("screen_shot_" + std::to_string(nScreenShotCount++) +".ppm");
                        screenShot << "P3\n" << std::to_string(ScreenHeight()) + " " + std::to_string(ScreenWidth()) + "\n" << "256\n";
                        for (int x = 0; x < ScreenWidth() - sideBarWidth; x++)
                        {
                            for (int y = 0; y < ScreenHeight(); y++)
                            {
                                int n = pFractalIterations[x * ScreenHeight() + y];
                                if (n == nMaxIteration)
                                    screenShot << 0 << " " << 0 << " " << 0 << std::endl;
                                else
                                {
                                    n = n % 255;
                                    screenShot << n << " " << n << " " << n << std::endl;
                                }
                            }
                        }
                        screenShot.close();
                        toggleScreenShot = false;
                    }
                    else
                    {
                        #pragma omp parallel for
                        for (int x = 0; x < ScreenWidth() - sideBarWidth; x++)
                        {
                            for (int y = 0; y < ScreenHeight(); y++)
                            {
                                int n = pFractalIterations[x * ScreenHeight() + y];
                                if (n == nMaxIteration)
                                    Draw(x, y, olc::Pixel(0,0,0));
                                else
                                    Draw(x, y, olc::Pixel(n,n,n));
                            }
                        }
                    } 
                    break;
                case 2:
                    if(toggleScreenShot)
                    {
                        std::ofstream screenShot("screen_shot_" + std::to_string(nScreenShotCount++) +".ppm");
                        screenShot << "P3\n" << std::to_string(ScreenHeight()) + " " + std::to_string(ScreenWidth()) + "\n" << "256\n";
                        for (int x = 0; x < ScreenWidth() - sideBarWidth; x++)
                        {
                            for (int y = 0; y < ScreenHeight(); y++)
                            {
                                int n = pFractalIterations[x * ScreenHeight() + y];
                                screenShot << (n*n)%255 << " " << (n)%255 << " " << (n*3)%255 << std::endl;
                            }
                        }
                        screenShot.close();
                        toggleScreenShot = false;
                    }
                    else
                    {
                        #pragma omp parallel for
                        for (int x = 0; x < ScreenWidth() - sideBarWidth; x++)
                        {
                            for (int y = 0; y < ScreenHeight(); y++)
                            {
                                int n = pFractalIterations[x * ScreenHeight() + y];
                                Draw(x, y, olc::Pixel(n*n, n, n*3, 255));
                            }
                        } 
                    }
                    break;
                case 3:
                    // color for flower
                    for (int x = 0; x < ScreenWidth() - sideBarWidth; x++) {
                        for (int y = 0; y < ScreenHeight(); y++) {
                            int n = pFractalIterations[x * ScreenHeight() + y];
                            double t = (double)n / nMaxIteration;

                            int r = (int)(255 * sin(0.036 * n + 1));
                            int g = (int)(255 * sin(0.036 * n + 2));
                            int b = (int)(255 * sin(0.036 * n + 3));

                            Draw(x, y, olc::Pixel(r, g, b));
                        }
                    }
                    break;
                case 4:
                    #pragma omp parallel for
                    for (int x = 0; x < ScreenWidth() - sideBarWidth; x++) {
                        for (int y = 0; y < ScreenHeight(); y++) {
                            int n = pFractalIterations[x * ScreenHeight() + y];
                            double t = (double)n / (double)nMaxIteration;
                            int r = (int)(127.5 * (1 + sin(10 * t)));
                            int g = (int)(127.5 * (1 + sin(10 * t + 2)));
                            int b = (int)(127.5 * (1 + sin(10 * t + 4)));
                            Draw(x, y, olc::Pixel(r, g, b));
                        }
                    }
                    break;
            }
        }
        
        if(nFracMode == 0)
        {
            calcName = "Sequencial";
            color = olc::CYAN;
            fracModeColor[0] = olc::CYAN;
            fracModeColor[1] = MENU_ITEM;
            fracModeColor[2] = MENU_ITEM;
        }
        else if (nFracMode == 1)
        {
            calcName = "Paralelo";
            color = olc::YELLOW;
            fracModeColor[0] = MENU_ITEM;
            fracModeColor[1] = olc::YELLOW;
            fracModeColor[2] = MENU_ITEM;
        }
        else if (nFracMode == 2)
        {
            calcName = "Paralelo c/ Ins. Vetoriais.";
            color = olc::GREEN;
            fracModeColor[0] = MENU_ITEM;
            fracModeColor[1] = MENU_ITEM;
            fracModeColor[2] = olc::GREEN;
        }

        if(nColorMode == 0)
        {
            fracCol = fracColorCol[0] = olc::Pixel(126, 156, 247);
            fracColorCol[1] = MENU_ITEM;
            fracColorCol[2] = MENU_ITEM;
        }
        else if(nColorMode == 1)
        {
            fracColorCol[0] = MENU_ITEM;
            fracCol = fracColorCol[1] = olc::DARK_GREY;
            fracColorCol[2] = MENU_ITEM;
        }
        else if(nColorMode == 2)
        {
            fracColorCol[0] = MENU_ITEM;
            fracColorCol[1] = MENU_ITEM;
            fracCol = fracColorCol[2] = olc::RED;
        }

        constexpr int uiDist = 25;

        if(toggleBenchmark)
        {
            switch (fracType)
            {
                case 0:
                    DrawString(5, 10,  "Fractal: Mandelbrot", fracCol, 2);
                    break;
                case 1:
                    DrawString(5, 10,  "Fractal: Flor de Mandelbrot", fracCol, 2);
                    break;
                case 2:
                    DrawString(5, 10,  "Fractal: Flor de Julia", fracCol, 2);
                    break;
                case 3:
                    DrawString(5, 10,  "Fractal: L-System", fracCol, 2);
                    break;
            }
            DrawString(5, 35,  "Iteracoes: " + std::to_string(nMaxIteration), colorIterations, 2);
            DrawString(5, 60, "Tempo: " + std::to_string(fTime.count()) + "s", color, 2);
            DrawString(5, 85, "FPS: " + std::to_string(1/fTime.count()) + "", color, 2);
            DrawString(5, 110, "[P] Tirar PrintScreen", colorIterations, 1);
            DrawString(5, 120, "[S] Ocultar Menu", colorIterations, 1);
            
            if (fracType != 3){
                DrawString(5, 130, "[F1/F2/F3/F4/F5] Alterar Cores", colorIterations, 1);
            } else {
                DrawString(5, 130, "Base: " + lsystemAxiom, colorIterations, 1);
                DrawString(5, 140, "Regras: " + rulesToString(lsystemRules), colorIterations, 1);
            }
        }
        
        //HUD
        int heightChange = 0;
        FillRect(ScreenWidth() - sideBarWidth, 0, sideBarWidth, ScreenHeight(), olc::VERY_DARK_BLUE);
        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "Controles:" , MENU_TITLE, 2); heightChange += 1;
        if (fracType != 3) {
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  Clique e arraste com" , mouseColor, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "o mouse para mover" , mouseColor, 2); heightChange += 1;

        }
        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  E - Zoom In" , zoomColor[0], 2); heightChange += 1;
        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  Q - Zoom Out" , zoomColor[1], 2); heightChange += 1;
        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  + - Mais detalhes" , MENU_ITEM, 2); heightChange += 1;
        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  - - Menos detalhes" , MENU_ITEM, 2); heightChange += 1;
        
        if (fracType == 1) {
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  CIMA - Aumenta o num." , MENU_ITEM, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  de petalas" , MENU_ITEM, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  BAIXO - Diminui o num." , MENU_ITEM, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  de petalas" , MENU_ITEM, 2); heightChange += 1;
        }
        
        if (fracType == 2) {
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  Setas - Alterar o" , MENU_ITEM, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  fractal" , MENU_ITEM, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  K - Mudar Preset" , MENU_ITEM, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  R - Resetar" , MENU_ITEM, 2); heightChange += 1;
        }

        if (fracType == 3) {
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  DIREITA/ESQUERDA - " , MENU_ITEM, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  Alterar o ângulo" , MENU_ITEM, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  K - Mudar Preset" , MENU_ITEM, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  I - Inserir Valor", MENU_ITEM, 2); heightChange += 1;
            if (isTyping) {
                DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "Voce esta digitando", olc::RED, 2); heightChange += 1;
                if (inputStage == InputStage::Rules) {
                    DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "Digite uma regra", olc::WHITE, 2);heightChange += 1;
                    DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "(ex: F=FF)", olc::WHITE, 2);heightChange += 1;
                    DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, currentRule, olc::YELLOW, 2);heightChange += 1;
                    DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "[ENTER] para confirmar regra, [R] para terminar", olc::DARK_YELLOW, 1);heightChange += 1;
                } else if (inputStage == InputStage::Axiom) {
                    DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "Digite a base:", olc::WHITE, 2);heightChange += 1;
                    DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, axiom, olc::YELLOW, 2);heightChange += 1;
                    DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "[ENTER] para confirmar", olc::VERY_DARK_YELLOW, 1);heightChange += 1;
                }
                
            }
        }

        if (fracType != 3) {
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "Mudar o modelo de" , MENU_TITLE, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "programacao:" , MENU_TITLE, 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  1 - Sequencial" , fracModeColor[0], 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  2 - Paralelo" , fracModeColor[1], 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  3 - Paralelo com" , fracModeColor[2], 2); heightChange += 1;
            DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "Instruc. Vetoriais" , fracModeColor[2], 2); heightChange += 1;
        }

        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "Trocar Fractal:", MENU_TITLE, 2); heightChange += 1;
        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  Z - Mandelbrot", MENU_ITEM, 2); heightChange += 1;
        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  X - Flor Mandelbrot", MENU_ITEM, 2); heightChange += 1;
        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  C - Flor de Julia", MENU_ITEM, 2); heightChange += 1;
        DrawString(ScreenWidth() - sideBarWidth + 10, 10 + uiDist * heightChange, "  V - L-System", MENU_ITEM, 2); heightChange += 1;

        //DrawString(ScreenWidth() - 660, ScreenHeight() - 30, "Mandel2Us - github.com/lucaszm7/Mandel2Us", olc::Pixel(255,255,255,123), 2);
        
        
        //std::cout << "Retornando true" << std::endl;
        return true;
	}

    ~MandelbrotFractal()
    {
        delete[] pFractalIterations;
        for(int i = 0; i < nNodesSize; ++i)
            delete[] pNodesParam[i];
        delete[] pNodesParam;
        setenv("SCREENSHOTCOUNT", std::to_string(nScreenShotCount).c_str(), 1);
    }

    std::string getTyping() {
        std::cout << "Digite e Pressione Enter: ";
        std::string s;
        std::getline(std::cin, s);
        return s;
    }

    std::string rulesToString(const std::map<char, std::string>& rules) {
        std::string result;
        for (const auto& [symbol, replacement] : rules)
        {
            result += symbol;
            result += " -> ";
            result += replacement;
            result += "\n";
        }
        return result;
    }

    void TakeScreenshot()
    {
        olc::Sprite* spr = GetDrawTarget();
        if (!spr) return;

        // Gera nome com timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
    #ifdef _WIN32
        localtime_s(&tm, &t);
    #else
        localtime_r(&t, &tm);
    #endif

        char filename[128];
        std::strftime(filename, sizeof(filename), "screenshot_%Y-%m-%d_%H-%M-%S.ppm", &tm);

        // Salva em formato PPM (fácil de abrir em qualquer editor de imagem)
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        file << "P6\n" << spr->width - sideBarWidth  << " " << spr->height << "\n255\n";

        for (int y = 0; y < spr->height; y++) {
            for (int x = 0; x < spr->width - sideBarWidth; x++) {
                olc::Pixel p = spr->GetPixel(x, y);
                file.put(p.r);
                file.put(p.g);
                file.put(p.b);
            }
        }
        file.close();

        std::cout << "[OK] Screenshot salva em: " << filename << std::endl;
    }

// Pan and Zoom Created with help of tutorials on channel @OneLoneCoder
protected:
    // Pan & Zoom variables
	olc::vd2d vOffset = { 0.0, 0.0 };
	olc::vd2d vStartPan = { 0.0, 0.0 };
	olc::vd2d vScale = { 1.0, 1.0 };

    void ScreenToWorld(const olc::vi2d& n, olc::vd2d& v)
	{
		v.x = (double)(n.x) / vScale.x + vOffset.x;
		v.y = (double)(n.y) / vScale.y + vOffset.y;
	}

    // Converte coords from Screen Space to World Space
    void ScreenToFrac(const olc::vi2d& screen_tl_before, const olc::vi2d& screen_br_before, 
                           const olc::vd2d& world_tl_before, const olc::vd2d& world_br_before, 
                           olc::vd2d& world_real_after, olc::vd2d& world_imag_after)
    {
        olc::vd2d screen_tl_after;
        ScreenToWorld(screen_tl_before, screen_tl_after);
        olc::vd2d screen_br_after;
        ScreenToWorld(screen_br_before, screen_br_after);
        
        world_real_after.x = map(screen_tl_after.x, (double)screen_tl_before.x, (double)screen_br_before.x, world_tl_before.x, world_br_before.x);
        world_real_after.y = map(screen_br_after.x, (double)screen_tl_before.x, (double)screen_br_before.x, world_tl_before.x, world_br_before.x);

        world_imag_after.x = map(screen_tl_after.y, (double)screen_tl_before.y, (double)screen_br_before.y, world_tl_before.y, world_br_before.y);
        world_imag_after.y = map(screen_br_after.y, (double)screen_tl_before.y, (double)screen_br_before.y, world_tl_before.y, world_br_before.y);

    }

    void ResetFractalView()
    {
        // Reset zoom e posição
        vScale = { 1.0, 1.0 };
        vOffset = { 0.0, 0.0 };
        vStartPan = { 0.0, 0.0 };

        // Reset Mandelbrot e Flower
        nMaxIteration = 32;
        nPetals = 5;
    }
};

int main(int argc, char** argv)
{
    int nScreenWidth  = 1500;
    int nScreenHeight = 750;
    int nMyRank, nNodesSize;

    if(!UseMPI) // Sequencial
    {
        MandelbrotFractal demo; // Constrói a janela
        if (demo.Construct(nScreenWidth, nScreenHeight, 1, 1, true, false))
            demo.Start(); // Inicia a janela
    }

    else // Paralelo
    {
        MPI::Init(argc, argv);

        nNodesSize = MPI::COMM_WORLD.Get_size();
        nMyRank = MPI::COMM_WORLD.Get_rank();
        
        if(nMyRank == 0) // Nó mestre
        {
            MandelbrotFractal demo; // Constrói a janela
            if (demo.Construct(nScreenWidth, nScreenHeight, 1, 1, false, false))
                demo.Start(); // Inicia a janela

            // Essa parte cuida da finalização dos outros nós
            // O sinal de finalização é um array de 10 doubles, onde o último elemento é -1
            double** pFinishCode;
            pFinishCode = new double*[nNodesSize];
            for(int i = 0; i < (nNodesSize - 1); ++i) // -1 pois o nó mestre não precisa de finalização
            {
                pFinishCode[i] = new double[10];
                pFinishCode[i][9] = -1.0; // Sinal de finalização, último elemento do array recebe -1
                MPI::COMM_WORLD.Send((void*)pFinishCode[i], 10, MPI::DOUBLE, i+1, 0); // Envia o sinal de finalização, i+1 para evitar enviar ao nodo mestre
                delete[] pFinishCode[i];
            }
            delete[] pFinishCode;

            std::cout << "I node " << nMyRank << " have finish!\n";
        }

        // Other nodes just do the computation
        else
        {
            double pParam[10]{0};
            int* pFractalIterations = new int[nScreenWidth * nScreenHeight]{0};
            
            while(pParam[9] >= 0) // enquanto não receber o array com -1 na posição 9, continua computando
            {
                // Receive
                MPI::COMM_WORLD.Recv((void*)pParam, 10, MPI::DOUBLE, 0, MPI::ANY_TAG);
                if(pParam[9] == -1) // recebeu o sinal de finalização
                    break;
                
                // Compute
                CreateMandelbrotParallelAVX({(int)pParam[0], (int)pParam[1]}, {(int)pParam[2], (int)pParam[3]}, 
                            {pParam[4], pParam[5]}, {pParam[6], pParam[7]}, 
                            pFractalIterations, (int)pParam[8], (int)pParam[3]);
                /* pParam:
                0 - starting x
                1 - starting y
                2 - width
                3 - height
                4 - real part top-left corner
                5 - imag part top-left corner
                6 - real part bottom-right corner
                7 - imag part bottom-right corner
                8 - nMaxIteration
                pFraclalIterations - array de int onde será armazenado o resultado
                */
                // Send Back
                MPI::COMM_WORLD.Send((void*)(pFractalIterations + ((int)pParam[0] * (int)pParam[3])), ((nScreenHeight*nScreenWidth)/nNodesSize), MPI::INT, 0, 0);
            }

            delete[] pFractalIterations;
            std::cout << "I node " << nMyRank << " have finish!\n";
        }

        std::cout << "I node " << nMyRank << " am waiting for other nodes to finish...\n";
        MPI::COMM_WORLD.Barrier(); // Sincronização
        MPI::Finalize();
        std::cout << "All nodes have finished!\n";
    }
    return 0;
}
