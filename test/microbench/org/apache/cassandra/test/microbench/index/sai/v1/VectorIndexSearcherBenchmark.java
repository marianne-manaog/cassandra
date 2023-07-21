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
package org.apache.cassandra.test.microbench.index.sai.v1;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.cassandra.SchemaLoader;
import org.apache.cassandra.config.Config;
import org.apache.cassandra.config.DatabaseDescriptor;
import org.apache.cassandra.cql3.CQLTester;
import org.apache.cassandra.cql3.Operator;
import org.apache.cassandra.cql3.QueryProcessor;
import org.apache.cassandra.db.ColumnFamilyStore;
import org.apache.cassandra.db.PartitionPosition;
import org.apache.cassandra.db.commitlog.CommitLog;
import org.apache.cassandra.db.compaction.CompactionManager;
import org.apache.cassandra.db.marshal.FloatType;
import org.apache.cassandra.db.marshal.VectorType;
import org.apache.cassandra.dht.AbstractBounds;
import org.apache.cassandra.dht.Murmur3Partitioner;
import org.apache.cassandra.gms.Gossiper;
import org.apache.cassandra.index.sai.IndexContext;
import org.apache.cassandra.index.sai.SSTableContext;
import org.apache.cassandra.index.sai.SSTableQueryContext;
import org.apache.cassandra.index.sai.StorageAttachedIndex;
import org.apache.cassandra.index.sai.StorageAttachedIndexGroup;
import org.apache.cassandra.index.sai.disk.format.IndexDescriptor;
import org.apache.cassandra.index.sai.disk.v1.IndexSearcher;
import org.apache.cassandra.index.sai.disk.v1.V1SearchableIndex;
import org.apache.cassandra.index.sai.plan.Expression;
import org.apache.cassandra.index.sai.utils.PrimaryKey;
import org.apache.cassandra.index.sai.utils.RangeIterator;
import org.apache.cassandra.index.sai.utils.SaiRandomizedTest;
import org.apache.cassandra.io.sstable.format.SSTableReader;
import org.apache.cassandra.schema.KeyspaceParams;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import static org.apache.cassandra.index.sai.LongVectorTest.randomVector;
import static org.apache.cassandra.index.sai.SAITester.waitForIndexQueryable;

@Fork(1)
@Warmup(time = 1, iterations = 1)
@Measurement(time = 1, iterations = 2)
@OutputTimeUnit(TimeUnit.SECONDS)
@BenchmarkMode(Mode.Throughput)
@State(Scope.Thread)
public class VectorIndexSearcherBenchmark extends SaiRandomizedTest
{
    private static final String KEYSPACE = "ks";
    private static final String TABLE = "tab";
    private static final int DIMENSION = 16;
    private static final int N = 10_000;
    private IndexSearcher searcher;
    private IndexContext indexContext;
    private AbstractBounds<PartitionPosition> bounds;

    @Setup(Level.Trial)
    public void setupCQLTester()
    {
        DatabaseDescriptor.daemonInitialization(() -> {
            Config config = DatabaseDescriptor.loadConfig();
            config.partitioner = Murmur3Partitioner.class.getName();
            return config;
        });

        SchemaLoader.prepareServer();
        Gossiper.instance.maybeInitializeLocalState(0);
        SchemaLoader.createKeyspace(KEYSPACE, KeyspaceParams.simple(1));
        QueryProcessor.executeInternal(String.format("CREATE TABLE %s.%s (key int primary key, value vector<float, %s>)", KEYSPACE, TABLE, DIMENSION));
        QueryProcessor.executeInternal(String.format("CREATE CUSTOM INDEX ON %s.%s(value) USING 'StorageAttachedIndex' WITH OPTIONS = { 'similarity_function': 'dot_product' }", KEYSPACE, TABLE));
        waitForIndexQueryable(KEYSPACE, TABLE);
        var keys = IntStream.range(0, N).boxed().collect(Collectors.toList());
        Collections.shuffle(keys);
        for (int i = 0; i < N; i++)
        {
            QueryProcessor.executeInternal(String.format("INSERT INTO %s.%s (key, value) VALUES (?, ?)", KEYSPACE, TABLE), i, randomVector(DIMENSION));
        }
        ColumnFamilyStore cfs = ColumnFamilyStore.getIfExists(KEYSPACE, TABLE);
        cfs.forceBlockingFlush(ColumnFamilyStore.FlushReason.UNIT_TESTS);

        StorageAttachedIndex sai = (StorageAttachedIndex) cfs.getIndexManager().getIndexByName(String.format("%s_value_idx", TABLE));
        indexContext = sai.getIndexContext();

        SSTableReader reader = cfs.getLiveSSTables().iterator().next();
        SSTableContext ctx = StorageAttachedIndexGroup.getIndexGroup(cfs).sstableContextManager().getContext(reader);
        V1SearchableIndex searchableIndex = (V1SearchableIndex) IndexDescriptor.create(cfs.getLiveSSTables().iterator().next()).newSearchableIndex(ctx, indexContext);

        bounds = AbstractBounds.bounds(reader.first, true, reader.last, true);
        searcher = searchableIndex.segments.get(0).index;
    }

    private ByteBuffer randomVectorBytes()
    {
        return VectorType.getInstance(FloatType.instance, DIMENSION).getSerializer().serialize(randomVector(DIMENSION));
    }

    @Benchmark
    public void doTestEqQueriesAgainstStringIndex() throws Exception
    {
        Expression expression = new Expression(indexContext).add(Operator.ANN, randomVectorBytes());
        RangeIterator<PrimaryKey> results = searcher.search(expression, bounds, SSTableQueryContext.forTest(), false, 1);
    }

    @TearDown(Level.Trial)
    public void teardown() throws IOException, InterruptedException
    {
        CompactionManager.instance.forceShutdown();
        CommitLog.instance.shutdownBlocking();
        CQLTester.cleanup();
    }

    public static void main(String[] args) throws Exception
    {
        VectorIndexSearcherBenchmark benchmark = new VectorIndexSearcherBenchmark();
        benchmark.setupCQLTester();
        benchmark.doTestEqQueriesAgainstStringIndex();
        benchmark.teardown();
    }
}