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
package org.apache.cassandra.db;

import java.util.Collection;
import java.util.UUID;

import io.reactivex.Observable;
import io.reactivex.Single;
import org.apache.cassandra.db.partitions.PartitionUpdate;

public interface IMutation
{
    public void apply();
    public Single<Integer> applyAsync();
    public String getKeyspaceName();
    public Collection<UUID> getColumnFamilyIds();
    public DecoratedKey key();
    public long getTimeout();
    public String toString(boolean shallow);
    public Collection<PartitionUpdate> getPartitionUpdates();

    /**
     * Computes the total data size of the specified mutations.
     * @param mutations the mutations
     * @return the total data size of the specified mutations
     */
    public static long dataSize(Collection<? extends IMutation> mutations)
    {
        long size = 0;
        for (IMutation mutation : mutations)
        {
            for (PartitionUpdate update : mutation.getPartitionUpdates())
                size += update.dataSize();
        }
        return size;
    }
}
