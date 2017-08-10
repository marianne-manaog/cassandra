/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.cassandra.utils;

import java.util.concurrent.TimeUnit;

import com.google.common.util.concurrent.Uninterruptibles;

import org.apache.cassandra.db.monitoring.ApproximateTime;

/**
 * Time source backed by {@link org.apache.cassandra.db.monitoring.ApproximateTime}.
 */
public class ApproximateTimeSource implements TimeSource
{
    @Override
    public long currentTimeMillis()
    {
        return ApproximateTime.currentTimeMillis();
    }

    @Override
    public long nanoTime()
    {
        return ApproximateTime.nanoTime();
    }
    
    @Override
    public void autoAdvance(int calls, long time, TimeUnit unit)
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public TimeSource sleepUninterruptibly(long sleepFor, TimeUnit unit)
    {
        Uninterruptibles.sleepUninterruptibly(sleepFor, unit);
        return this;
    }

    @Override
    public TimeSource sleep(long sleepFor, TimeUnit unit) throws InterruptedException
    {
        TimeUnit.NANOSECONDS.sleep(TimeUnit.NANOSECONDS.convert(sleepFor, unit));
        return this;
    }
}
