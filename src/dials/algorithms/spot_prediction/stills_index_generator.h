/*
 * stills_index_generator.h
 *
 *  Copyright (C) 2013 Diamond Light Source, CCP4
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_SPOT_PREDICTION_STILLS_INDEX_GENERATOR_H
#define DIALS_ALGORITHMS_SPOT_PREDICTION_STILLS_INDEX_GENERATOR_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <cctbx/miller.h>
#include <cctbx/sgtbx/space_group_type.h>
#include <cctbx/uctbx.h>
#include <scitbx/mat3.h>
#include <scitbx/vec3.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/spot_prediction/index_generator.h>
#include <dials/error.h>

namespace dials { namespace algorithms {

  using scitbx::mat3;
  using scitbx::vec3;

  /**
   * Generate the Miller indices a still image can diffract.
   *
   * A reflection diffracts when its reciprocal lattice point q = ub h lies on the
   * Ewald sphere. The offset from it,
   *
   *     eps(h) = q.q + 2 q.s0,
   *
   * is a quadratic form in h, so the indices with |eps| <= eps_cut occupy a thin
   * ellipsoidal shell, and those within the resolution limit occupy an ellipsoid
   * concentric with the origin. Walking the intersection directly costs about three
   * orders of magnitude less than testing every point of the bounding box.
   *
   * Indices are emitted in the same order as IndexGenerator - h ascending, then k,
   * then l - and carry the same resolution and systematic absence tests, so the
   * result is a subsequence of what IndexGenerator produces. The shell bounds are
   * rounded outwards, so a few indices beyond the cutoff may be emitted; the caller
   * applies the exact test. Where eps_cut is not finite, or ub is degenerate, every
   * index within the resolution limit is emitted.
   */
  class StillsIndexGenerator {
  public:
    typedef cctbx::miller::index<> miller_index;

    StillsIndexGenerator() : exhausted_(true), walk_shell_(false) {}

    /**
     * Initialise the generator.
     * @param unit_cell The unit cell structure
     * @param space_group_type The space group type structure
     * @param d_min The resolution limit
     * @param ub The UB matrix
     * @param s0 The incident beam vector
     * @param eps_cut The largest |q.q + 2 q.s0| worth considering
     */
    StillsIndexGenerator(cctbx::uctbx::unit_cell const& unit_cell,
                         cctbx::sgtbx::space_group_type const& space_group_type,
                         double d_min,
                         mat3<double> const& ub,
                         vec3<double> const& s0,
                         double eps_cut)
        : unit_cell_(unit_cell),
          space_group_type_(space_group_type),
          d_min_(d_min),
          eps_cut_(eps_cut),
          exhausted_(false),
          walk_shell_(false) {
      DIALS_ASSERT(d_min > 0.0);
      if (!(eps_cut_ > 0.0) || !is_finite(eps_cut_) || !decompose(ub, s0)) {
        fallback_ = IndexGenerator(unit_cell, space_group_type, d_min);
        return;
      }
      walk_shell_ = true;
      if (!start_h()) {
        exhausted_ = true;
      }
    }

    /**
     * Get the next Miller index, or (0, 0, 0) once there are none left.
     * @returns The next miller index
     */
    miller_index next() {
      if (!walk_shell_) {
        return fallback_.next();
      }
      while (!exhausted_) {
        miller_index h(h_, k_, l_);
        if (!advance()) {
          exhausted_ = true;
        }
        if (unit_cell_.d(h) >= d_min_ && !space_group_type_.group().is_sys_absent(h)) {
          return h;
        }
      }
      return miller_index(0, 0, 0);
    }

    /**
     * Create an array of Miller indices by calling next until (000) is reached.
     * @returns The array of valid miller indices
     */
    af::shared<miller_index> to_array() {
      af::shared<miller_index> result;
      for (;;) {
        miller_index h = this->next();
        if (h.is_zero()) {
          break;
        }
        result.push_back(h);
      }
      return result;
    }

  private:
    static bool is_finite(double x) {
      return x == x && x != std::numeric_limits<double>::infinity()
             && x != -std::numeric_limits<double>::infinity();
    }

    /**
     * Factorise eps into nested squares, so that the range of each index given the
     * ones outside it is a quadratic inequality:
     *
     *   eps    = d3 (l + m13 h + m23 k + q3)^2 + shell_hk(h, k)
     *   q.q    = d3 (l + m13 h + m23 k)^2      + resolution_hk(h, k)
     *
     * and likewise one level up. Returns false if ub is degenerate.
     */
    bool decompose(mat3<double> const& ub, vec3<double> const& s0) {
      vec3<double> a1(ub[0], ub[3], ub[6]);
      vec3<double> a2(ub[1], ub[4], ub[7]);
      vec3<double> a3(ub[2], ub[5], ub[8]);

      double g11 = a1 * a1, g22 = a2 * a2, g33 = a3 * a3;
      double g12 = a1 * a2, g13 = a1 * a3, g23 = a2 * a3;
      double p1 = a1 * s0, p2 = a2 * s0, p3 = a3 * s0;

      if (!(g33 > 0.0)) return false;
      d3_ = g33;
      m13_ = g13 / g33;
      m23_ = g23 / g33;
      q3_ = p3 / g33;

      double g22a = g22 - g23 * m23_;
      double g12a = g12 - g13 * m23_;
      double g11a = g11 - g13 * m13_;
      double p2a = p2 - p3 * m23_;
      double p1a = p1 - p3 * m13_;
      double c_a = -p3 * q3_;

      if (!(g22a > 0.0)) return false;
      d2_ = g22a;
      m12_ = g12a / g22a;
      q2_ = p2a / g22a;

      double g11b = g11a - g12a * m12_;
      double p1b = p1a - p2a * m12_;
      double c_b = c_a - p2a * q2_;

      if (!(g11b > 0.0)) return false;
      d1_ = g11b;
      q1_ = p1b / d1_;
      c1_ = c_b;

      q_max_sq_ = 1.0 / (d_min_ * d_min_);
      // the resolution limit is enforced exactly on each emitted index, so a little
      // slack here only costs a few extra candidates
      q_max_sq_ *= 1.0 + 1e-6;
      return true;
    }

    /// Integers x with (x - centre)^2 <= half_width_sq, rounded outwards.
    static bool integer_range(double centre, double half_width_sq, int& lo, int& hi) {
      if (!(half_width_sq >= 0.0)) return false;
      double half_width = std::sqrt(half_width_sq) + 1e-6;
      lo = (int)std::ceil(centre - half_width);
      hi = (int)std::floor(centre + half_width);
      return lo <= hi;
    }

    /// min over real k and l of eps, at this h
    double shell_h(double h) const {
      return d1_ * (h + q1_) * (h + q1_) - d1_ * q1_ * q1_ + c1_;
    }

    /// min over real l of eps, at this h and k
    double shell_hk(double h, double k) const {
      double t = k + m12_ * h + q2_;
      return d2_ * t * t + shell_h(h);
    }

    /// min over real l of q.q, at this h and k
    double resolution_hk(double h, double k) const {
      double t = k + m12_ * h;
      return d2_ * t * t + d1_ * h * h;
    }

    bool start_h() {
      // eps has its minimum over all real indices at h = -q1_
      double c_min = c1_ - d1_ * q1_ * q1_;
      int shell_lo, shell_hi, res_lo, res_hi;
      if (!integer_range(-q1_, (eps_cut_ - c_min) / d1_, shell_lo, shell_hi))
        return false;
      if (!integer_range(0.0, q_max_sq_ / d1_, res_lo, res_hi)) return false;
      h_lo_ = std::max(shell_lo, res_lo);
      h_hi_ = std::min(shell_hi, res_hi);
      for (h_ = h_lo_; h_ <= h_hi_; ++h_) {
        if (start_k()) return true;
      }
      return false;
    }

    bool start_k() {
      double h = h_;
      int shell_lo, shell_hi, res_lo, res_hi;
      if (!integer_range(
            -(m12_ * h + q2_), (eps_cut_ - shell_h(h)) / d2_, shell_lo, shell_hi)) {
        return false;
      }
      if (!integer_range(-m12_ * h, (q_max_sq_ - d1_ * h * h) / d2_, res_lo, res_hi)) {
        return false;
      }
      k_lo_ = std::max(shell_lo, res_lo);
      k_hi_ = std::min(shell_hi, res_hi);
      for (k_ = k_lo_; k_ <= k_hi_; ++k_) {
        if (start_l()) return true;
      }
      return false;
    }

    bool start_l() {
      double h = h_, k = k_;
      double shell_centre = -(m13_ * h + m23_ * k + q3_);
      double offset = shell_hk(h, k);

      int shell_lo, shell_hi;
      if (!integer_range(shell_centre, (eps_cut_ - offset) / d3_, shell_lo, shell_hi)) {
        return false;
      }
      int res_lo, res_hi;
      if (!integer_range(-(m13_ * h + m23_ * k),
                         (q_max_sq_ - resolution_hk(h, k)) / d3_,
                         res_lo,
                         res_hi)) {
        return false;
      }

      // where the l line passes inside the shell, the indices between the two
      // crossings are too far from the Ewald sphere to diffract
      int lo[2], hi[2];
      int n = 1;
      lo[0] = shell_lo;
      hi[0] = shell_hi;
      double gap_sq = (-eps_cut_ - offset) / d3_;
      if (gap_sq > 0.0) {
        double gap = std::sqrt(gap_sq) - 1e-6;
        if (gap > 0.0) {
          int gap_hi = (int)std::floor(shell_centre - gap);
          int gap_lo = (int)std::ceil(shell_centre + gap);
          if (gap_lo > gap_hi + 1) {
            n = 2;
            hi[0] = std::min(shell_hi, gap_hi);
            lo[1] = std::max(shell_lo, gap_lo);
            hi[1] = shell_hi;
          }
        }
      }

      n_runs_ = 0;
      for (int i = 0; i < n; ++i) {
        int run_lo = std::max(lo[i], res_lo);
        int run_hi = std::min(hi[i], res_hi);
        if (run_lo <= run_hi) {
          run_lo_[n_runs_] = run_lo;
          run_hi_[n_runs_] = run_hi;
          ++n_runs_;
        }
      }
      if (n_runs_ == 0) return false;
      run_ = 0;
      l_ = run_lo_[0];
      return true;
    }

    /// Step to the next candidate, returning false once there are none left.
    bool advance() {
      if (l_ < run_hi_[run_]) {
        ++l_;
        return true;
      }
      if (run_ + 1 < n_runs_) {
        ++run_;
        l_ = run_lo_[run_];
        return true;
      }
      for (++k_; k_ <= k_hi_; ++k_) {
        if (start_l()) return true;
      }
      for (++h_; h_ <= h_hi_; ++h_) {
        if (start_k()) return true;
      }
      return false;
    }

    cctbx::uctbx::unit_cell unit_cell_;
    cctbx::sgtbx::space_group_type space_group_type_;
    double d_min_;
    double eps_cut_;
    bool exhausted_;
    bool walk_shell_;
    IndexGenerator fallback_;

    double d1_, d2_, d3_;
    double m12_, m13_, m23_;
    double q1_, q2_, q3_;
    double c1_;
    double q_max_sq_;

    int h_, k_, l_;
    int h_lo_, h_hi_, k_lo_, k_hi_;
    int run_, n_runs_;
    int run_lo_[2], run_hi_[2];
  };

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_SPOT_PREDICTION_STILLS_INDEX_GENERATOR_H
