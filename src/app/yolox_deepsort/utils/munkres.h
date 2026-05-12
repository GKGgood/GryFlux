#if !defined(_MUNKRES_H_)
#define _MUNKRES_H_

#include "matrix.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef XYZMIN
#define XYZMIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef XYZMAX
#define XYZMAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

template <typename Data>
class Munkres
{
    static constexpr int NORMAL = 0;
    static constexpr int STAR = 1;
    static constexpr int PRIME = 2;

public:
    void solve(Matrix<Data> &m)
    {
        const size_t rows = m.rows();
        const size_t columns = m.columns();
        const size_t size = XYZMAX(rows, columns);

        matrix_ = m;

        if (rows != columns)
        {
            matrix_.resize(size, size, matrix_.mmax());
        }

        mask_matrix_.resize(size, size);

        row_mask_ = new bool[size];
        col_mask_ = new bool[size];
        for (size_t i = 0; i < size; ++i)
        {
            row_mask_[i] = false;
            col_mask_[i] = false;
        }

        replace_infinites(matrix_);
        minimize_along_direction(matrix_, rows >= columns);
        minimize_along_direction(matrix_, rows < columns);

        int step = 1;
        while (step)
        {
            switch (step)
            {
            case 1:
                step = step1();
                break;
            case 2:
                step = step2();
                break;
            case 3:
                step = step3();
                break;
            case 4:
                step = step4();
                break;
            case 5:
                step = step5();
                break;
            }
        }

        for (size_t row = 0; row < size; ++row)
        {
            for (size_t col = 0; col < size; ++col)
            {
                if (mask_matrix_(row, col) == STAR)
                {
                    matrix_(row, col) = 0;
                }
                else
                {
                    matrix_(row, col) = -1;
                }
            }
        }

        matrix_.resize(rows, columns);
        m = matrix_;

        delete[] row_mask_;
        delete[] col_mask_;
        row_mask_ = nullptr;
        col_mask_ = nullptr;
    }

    static void replace_infinites(Matrix<Data> &matrix)
    {
        const size_t rows = matrix.rows();
        const size_t columns = matrix.columns();
        double max_val = std::numeric_limits<Data>::lowest();
        bool found_finite = false;
        constexpr auto infinity = std::numeric_limits<Data>::infinity();

        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < columns; ++col)
            {
                if (matrix(row, col) != infinity && matrix(row, col) == matrix(row, col))
                {
                    if (!found_finite || matrix(row, col) > max_val)
                    {
                        max_val = matrix(row, col);
                        found_finite = true;
                    }
                }
            }
        }

        Data replacement_val;
        if (!found_finite)
        {
            replacement_val = 1;
        }
        else
        {
            if (max_val < std::numeric_limits<Data>::max())
            {
                replacement_val = static_cast<Data>(max_val + 1);
            }
            else
            {
                replacement_val = std::numeric_limits<Data>::max();
            }
            if (replacement_val == infinity)
            {
                replacement_val = static_cast<Data>(max_val);
            }
        }

        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < columns; ++col)
            {
                if (matrix(row, col) == infinity || matrix(row, col) != matrix(row, col))
                {
                    matrix(row, col) = replacement_val;
                }
            }
        }
    }

    static void minimize_along_direction(Matrix<Data> &matrix, const bool over_columns)
    {
        const size_t outer_size = over_columns ? matrix.columns() : matrix.rows();
        const size_t inner_size = over_columns ? matrix.rows() : matrix.columns();

        for (size_t i = 0; i < outer_size; ++i)
        {
            Data min_val = over_columns ? matrix(0, i) : matrix(i, 0);

            for (size_t j = 1; j < inner_size; ++j)
            {
                const Data current_val = over_columns ? matrix(j, i) : matrix(i, j);
                if (current_val < min_val)
                {
                    min_val = current_val;
                }
            }

            if (min_val > 0)
            {
                for (size_t j = 0; j < inner_size; ++j)
                {
                    if (over_columns)
                    {
                        matrix(j, i) -= min_val;
                    }
                    else
                    {
                        matrix(i, j) -= min_val;
                    }
                }
            }
        }
    }

private:
    inline bool find_uncovered_in_matrix(const Data item, size_t &row, size_t &col) const
    {
        const size_t rows = matrix_.rows();
        const size_t columns = matrix_.columns();
        for (row = 0; row < rows; ++row)
        {
            if (!row_mask_[row])
            {
                for (col = 0; col < columns; ++col)
                {
                    if (!col_mask_[col])
                    {
                        if constexpr (std::is_floating_point_v<Data>)
                        {
                            if (std::fabs(matrix_(row, col) - item) <
                                std::numeric_limits<Data>::epsilon())
                            {
                                return true;
                            }
                        }
                        else if (matrix_(row, col) == item)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    int step1()
    {
        const size_t rows = matrix_.rows();
        const size_t columns = matrix_.columns();
        std::vector<bool> row_starred(rows, false);
        std::vector<bool> col_starred(columns, false);

        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < columns; ++col)
            {
                bool is_zero = false;
                if constexpr (std::is_floating_point_v<Data>)
                {
                    is_zero =
                        std::fabs(matrix_(row, col)) < std::numeric_limits<Data>::epsilon();
                }
                else
                {
                    is_zero = matrix_(row, col) == 0;
                }

                if (is_zero && !row_starred[row] && !col_starred[col])
                {
                    mask_matrix_(row, col) = STAR;
                    row_starred[row] = true;
                    col_starred[col] = true;
                }
            }
        }
        return 2;
    }

    int step2()
    {
        const size_t rows = matrix_.rows();
        const size_t columns = matrix_.columns();
        size_t covercount = 0;
        for (size_t i = 0; i < columns; ++i)
        {
            col_mask_[i] = false;
        }

        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t col = 0; col < columns; ++col)
            {
                if (mask_matrix_(row, col) == STAR)
                {
                    col_mask_[col] = true;
                }
            }
        }

        for (size_t i = 0; i < columns; ++i)
        {
            if (col_mask_[i])
            {
                ++covercount;
            }
        }

        if (covercount >= matrix_.minsize())
        {
            return 0;
        }
        return 3;
    }

    int step3()
    {
        while (find_uncovered_in_matrix(0, saverow_, savecol_))
        {
            mask_matrix_(saverow_, savecol_) = PRIME;

            bool found_star_in_row = false;
            size_t star_col = 0;
            for (size_t ncol = 0; ncol < matrix_.columns(); ++ncol)
            {
                if (mask_matrix_(saverow_, ncol) == STAR)
                {
                    found_star_in_row = true;
                    star_col = ncol;
                    break;
                }
            }

            if (!found_star_in_row)
            {
                return 4;
            }

            row_mask_[saverow_] = true;
            col_mask_[star_col] = false;
        }

        return 5;
    }

    int step4()
    {
        const size_t rows = matrix_.rows();
        const size_t columns = matrix_.columns();
        std::list<std::pair<size_t, size_t>> seq;
        seq.emplace_back(saverow_, savecol_);

        size_t current_col = savecol_;

        while (true)
        {
            size_t star_row = static_cast<size_t>(-1);
            for (size_t r = 0; r < rows; ++r)
            {
                if (mask_matrix_(r, current_col) == STAR)
                {
                    star_row = r;
                    break;
                }
            }

            if (star_row == static_cast<size_t>(-1))
            {
                break;
            }

            seq.emplace_back(star_row, current_col);

            size_t prime_col = static_cast<size_t>(-1);
            for (size_t c = 0; c < columns; ++c)
            {
                if (mask_matrix_(star_row, c) == PRIME)
                {
                    prime_col = c;
                    break;
                }
            }

            seq.emplace_back(star_row, prime_col);
            current_col = prime_col;
        }

        for (const auto &pair : seq)
        {
            if (mask_matrix_(pair.first, pair.second) == STAR)
            {
                mask_matrix_(pair.first, pair.second) = NORMAL;
            }
            else
            {
                mask_matrix_(pair.first, pair.second) = STAR;
            }
        }

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < columns; ++c)
            {
                if (mask_matrix_(r, c) == PRIME)
                {
                    mask_matrix_(r, c) = NORMAL;
                }
            }
        }

        for (size_t i = 0; i < rows; ++i)
        {
            row_mask_[i] = false;
        }
        for (size_t i = 0; i < columns; ++i)
        {
            col_mask_[i] = false;
        }

        return 2;
    }

    int step5()
    {
        const size_t rows = matrix_.rows();
        const size_t columns = matrix_.columns();
        Data h = std::numeric_limits<Data>::max();
        bool found_uncovered = false;

        for (size_t row = 0; row < rows; ++row)
        {
            if (!row_mask_[row])
            {
                for (size_t col = 0; col < columns; ++col)
                {
                    if (!col_mask_[col] && matrix_(row, col) < h)
                    {
                        h = matrix_(row, col);
                        found_uncovered = true;
                    }
                }
            }
        }

        if (!found_uncovered || h == std::numeric_limits<Data>::max())
        {
            return 3;
        }

        for (size_t row = 0; row < rows; ++row)
        {
            if (row_mask_[row])
            {
                for (size_t col = 0; col < columns; ++col)
                {
                    matrix_(row, col) += h;
                }
            }
        }

        for (size_t col = 0; col < columns; ++col)
        {
            if (!col_mask_[col])
            {
                for (size_t row = 0; row < rows; ++row)
                {
                    matrix_(row, col) -= h;
                }
            }
        }

        return 3;
    }

    Matrix<int> mask_matrix_;
    Matrix<Data> matrix_;
    bool *row_mask_ = nullptr;
    bool *col_mask_ = nullptr;
    size_t saverow_ = 0;
    size_t savecol_ = 0;
};

#endif
