#ifndef SPINLOCK_H
#define SPINLOCK_H

// Required for the Spinlock definition
#include <atomic>
#include <thread>

// Include for _mm_pause() on x86/x64 systems
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#endif


// Macro to abstract the CPU pause instruction (x86/64)
#if defined(__GNUC__) || defined(__clang__)
#define CPU_PAUSE() __builtin_ia32_pause()
#elif defined(_MSC_VER)
#define CPU_PAUSE() _mm_pause()
#else
#define CPU_PAUSE()
#endif

struct Spinlock
{
    // A std::atomic_flag is the only guaranteed lock-free atomic primitive in C++11
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

    void lock()
    {
        uint_fast8_t spin_count = 0;

        // Waits in a loop (spins) until the flag is cleared, and then sets it atomically.
        while (flag.test_and_set(std::memory_order_acquire))
        {
            // OPTIMIZATION: Hybrid Backoff Strategy
            if (spin_count < 10) { // Fast spinning threshold (e.g., 10 iterations)
                // CPU Pause Instruction: Reduces power consumption and improves bus contention
                // during short busy-waits.
                CPU_PAUSE();
                spin_count++;
            } else {
                // If contention is high, yield the thread to the OS scheduler.
                // This releases the CPU core, preventing resource waste and improving overall system throughput.
                std::this_thread::yield();
            }
        }
    }

    void unlock()
    {
        // memory_order_release: Ensures that all changes become visible
        // before the lock is released.
        flag.clear(std::memory_order_release);
    }

    // Adds try_lock, for compatibility with std::lock_guard and other standard utilities
    bool try_lock() {
        return !flag.test_and_set(std::memory_order_acquire);
    }
};

#endif // SPINLOCK_H
