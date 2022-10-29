/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_MIOPEN_FIND_DB_HPP_
#define GUARD_MIOPEN_FIND_DB_HPP_

#include <miopen/config.h>
#include <miopen/db.hpp>
#include <miopen/db_path.hpp>
#include <miopen/db_record.hpp>
#include <miopen/env.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/ramdb.hpp>
#include <miopen/readonlyramdb.hpp>

#include <optional>

#include <functional>
#include <vector>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_DISABLE_FIND_DB)

namespace miopen {

struct Handle;
struct NetworkConfig;

template <class TDb>
class FindDbRecord_t;

#if MIOPEN_DEBUG_FIND_DB_CACHING
using SystemFindDb = ReadonlyRamDb;
using UserFindDb   = RamDb;
#else
using SystemFindDb = PlainTextDb;
using UserFindDb   = PlainTextDb;
#endif

using FindDb           = MultiFileDb<SystemFindDb, UserFindDb, false>;
using FindDbRecord     = FindDbRecord_t<FindDb>;
using UserFindDbRecord = FindDbRecord_t<UserFindDb>;

namespace debug {

// For unit tests.
extern bool testing_find_db_enabled; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)
extern std::optional<std::string>&
testing_find_db_path_override(); /// \todo Remove when #1723 is resolved.

} // namespace debug

bool CheckInvokerSupport(const std::string& algo);

template <class TDb>
class FindDbRecord_t
{
private:
    template <class TTestDb>
    using is_find_t = std::enable_if_t<std::is_same<TTestDb, UserFindDb>::value, int>;

    template <class TTestDb>
    using is_immediate_t = std::enable_if_t<std::is_same<TTestDb, FindDb>::value, int>;

public:
    FindDbRecord_t(const FindDbRecord_t&) = delete;
    FindDbRecord_t& operator=(const FindDbRecord_t&) = delete;

    template <class TProblemDescription, class TTestDb = TDb>
    FindDbRecord_t(Handle& handle, const TProblemDescription& problem, is_immediate_t<TTestDb> = 0)
        : path(debug::testing_find_db_path_override() ? *debug::testing_find_db_path_override()
                                                      : GetUserPath(handle)),
          installed_path(debug::testing_find_db_path_override()
                             ? *debug::testing_find_db_path_override()
                             : GetInstalledPath(handle)),
          db(debug::testing_find_db_enabled && !IsEnabled(MIOPEN_DEBUG_DISABLE_FIND_DB{})
                 ? std::make_optional<DbTimer<TDb>>(DbTimer<TDb>{installed_path, path})
                 : std::nullopt)
    {
        if(!db.has_value())
            return;

        content = db->FindRecord(problem);
        in_sync = content.has_value();
    }

    template <class TProblemDescription, class TTestDb = TDb>
    FindDbRecord_t(Handle& handle, const TProblemDescription& problem, is_find_t<TTestDb> = 0)
        : path(debug::testing_find_db_path_override() ? *debug::testing_find_db_path_override()
                                                      : GetUserPath(handle)),
#if MIOPEN_DISABLE_USERDB
          db(std::optional<DbTimer<TDb>>{})
#else
          db(debug::testing_find_db_enabled && !IsEnabled(MIOPEN_DEBUG_DISABLE_FIND_DB{})
                   ? std::make_optional<DbTimer<TDb>>(DbTimer<TDb>{path, false})
                   : std::nullopt)
#endif
    {
        if(!db.has_value())
            return;

        content = db->FindRecord(problem);
        in_sync = content.has_value();
    }

    ~FindDbRecord_t()
    {
        if(!db.has_value() || !content.has_value() || in_sync)
            return;
        if(!db->StoreRecord(content.value()))
            MIOPEN_LOG_E("Failed to store record to find-db at <" << path << ">");
    }

    auto begin() const { return content->As<FindDbData>().begin(); }
    auto begin() { return content->As<FindDbData>().begin(); }
    auto end() const { return content->As<FindDbData>().end(); }
    auto end() { return content->As<FindDbData>().end(); }
    bool empty() const { return !content.has_value(); }

    template <class TProblemDescription>
    static std::vector<PerfField> TryLoad(Handle& handle,
                                          const TProblemDescription& problem,
                                          const std::function<void(DbRecord&)>& regenerator)
    {
        auto ret = std::vector<PerfField>{};
        FindDbRecord_t<TDb> record{handle, problem};

        const auto network_config = problem.BuildConfKey();

        if(record.in_sync && !record.Validate(handle, network_config))
        {
            record.CopyTo(ret);
            return ret;
        }

        MIOPEN_LOG_I("Find-db regenerating.");
        ret.clear();
        record.in_sync = false;
        record.content.emplace(problem);
        regenerator(*record.content);
        record.CopyTo(ret);

        return ret;
    }

private:
    std::string path;
    std::string installed_path;
    std::optional<DbTimer<TDb>> db;
    std::optional<DbRecord> content{std::nullopt};
    bool in_sync = false;

    static bool HasKernel(Handle& handle, const FindDbKCacheKey& key);

    static std::string GetInstalledPath(Handle& handle);
    static std::string GetInstalledPathEmbed(Handle& handle);
    static std::string GetInstalledPathFile(Handle& handle);
    static std::string GetUserPath(Handle& handle);

    // Returns true if rebuild is required
    bool Validate(Handle& handle, const NetworkConfig& config) const;
    void CopyTo(std::vector<PerfField>& to) const;

    void LogFindDbItem(const std::pair<std::string, FindDbData>& pair,
                       bool log_as_error = false) const;
};

extern template class FindDbRecord_t<FindDb>;
extern template class FindDbRecord_t<UserFindDb>;

} // namespace miopen

#endif
