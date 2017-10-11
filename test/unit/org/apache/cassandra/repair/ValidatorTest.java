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
package org.apache.cassandra.repair;

import java.net.InetAddress;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.apache.cassandra.db.compaction.CompactionManager;
import org.apache.cassandra.db.compaction.CompactionsTest;
import org.apache.cassandra.io.sstable.format.SSTableReader;
import org.junit.After;
import org.junit.BeforeClass;
import org.junit.Test;

import org.apache.cassandra.SchemaLoader;
import org.apache.cassandra.net.interceptors.InterceptionContext;
import org.apache.cassandra.net.interceptors.Interceptor;
import org.apache.cassandra.net.Message;
import org.apache.cassandra.net.Verbs;
import org.apache.cassandra.schema.Schema;
import org.apache.cassandra.db.BufferDecoratedKey;
import org.apache.cassandra.db.ColumnFamilyStore;
import org.apache.cassandra.db.EmptyIterators;
import org.apache.cassandra.db.Keyspace;
import org.apache.cassandra.dht.IPartitioner;
import org.apache.cassandra.dht.Range;
import org.apache.cassandra.dht.Token;
import org.apache.cassandra.net.MessagingService;
import org.apache.cassandra.repair.messages.ValidationComplete;
import org.apache.cassandra.schema.KeyspaceParams;
import org.apache.cassandra.service.ActiveRepairService;
import org.apache.cassandra.streaming.PreviewKind;
import org.apache.cassandra.utils.ByteBufferUtil;
import org.apache.cassandra.utils.MerkleTree;
import org.apache.cassandra.utils.MerkleTrees;
import org.apache.cassandra.utils.FBUtilities;
import org.apache.cassandra.utils.UUIDGen;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public class ValidatorTest
{
    private static final long TEST_TIMEOUT = 60; //seconds

    private static final String keyspace = "ValidatorTest";
    private static final String columnFamily = "Standard1";
    private static IPartitioner partitioner;

    @BeforeClass
    public static void defineSchema() throws Exception
    {
        SchemaLoader.prepareServer();
        SchemaLoader.createKeyspace(keyspace,
                                    KeyspaceParams.simple(1),
                                    SchemaLoader.standardCFMD(keyspace, columnFamily));
        partitioner = Schema.instance.getTableMetadata(keyspace, columnFamily).partitioner;
    }

    @After
    public void tearDown()
    {
        MessagingService.instance().clearInterceptors();
    }

    @Test
    public void testValidatorComplete() throws Throwable
    {
        Range<Token> range = new Range<>(partitioner.getMinimumToken(), partitioner.getRandomToken());
        final RepairJobDesc desc = new RepairJobDesc(UUID.randomUUID(), UUID.randomUUID(), keyspace, columnFamily, Arrays.asList(range));

        final CompletableFuture<Message> outgoingMessageSink = registerOutgoingMessageSink();

        InetAddress remote = InetAddress.getByName("127.0.0.2");

        ColumnFamilyStore cfs = Keyspace.open(keyspace).getColumnFamilyStore(columnFamily);

        Validator validator = new Validator(desc, remote, 0, PreviewKind.NONE);
        MerkleTrees tree = new MerkleTrees(partitioner);
        tree.addMerkleTrees((int) Math.pow(2, 15), validator.desc.ranges);
        validator.prepare(cfs, tree);

        // and confirm that the tree was split
        assertTrue(tree.size() > 1);

        // add a row
        Token mid = partitioner.midpoint(range.left, range.right);
        validator.add(EmptyIterators.unfilteredRow(cfs.metadata(), new BufferDecoratedKey(mid, ByteBufferUtil.bytes("inconceivable!")), false));
        validator.complete();

        // confirm that the tree was validated
        Token min = tree.partitioner().getMinimumToken();
        assertNotNull(tree.hash(new Range<>(min, min)));

        Message<ValidationComplete> message = outgoingMessageSink.get(TEST_TIMEOUT, TimeUnit.SECONDS);
        assertEquals(Verbs.REPAIR, message.group());
        assertEquals(Verbs.REPAIR.VALIDATION_COMPLETE, message.verb());

        ValidationComplete vc = message.payload();

        assertEquals(desc, vc.desc);
        assertTrue(vc.success());
        assertNotNull(vc.trees);
    }


    @Test
    public void testValidatorFailed() throws Throwable
    {
        Range<Token> range = new Range<>(partitioner.getMinimumToken(), partitioner.getRandomToken());
        final RepairJobDesc desc = new RepairJobDesc(UUID.randomUUID(), UUID.randomUUID(), keyspace, columnFamily, Arrays.asList(range));

        final CompletableFuture<Message> outgoingMessageSink = registerOutgoingMessageSink();

        InetAddress remote = InetAddress.getByName("127.0.0.2");

        Validator validator = new Validator(desc, remote, 0, PreviewKind.NONE);
        validator.fail();

        Message<ValidationComplete> message = outgoingMessageSink.get(TEST_TIMEOUT, TimeUnit.SECONDS);
        assertEquals(Verbs.REPAIR, message.group());
        assertEquals(Verbs.REPAIR.VALIDATION_COMPLETE, message.verb());
        ValidationComplete vc = message.payload();
        assertEquals(desc, vc.desc);
        assertFalse(vc.success());
        assertNull(vc.trees);
    }

    @Test
    public void simpleValidationTest128() throws Exception
    {
        simpleValidationTest(128);
    }

    @Test
    public void simpleValidationTest1500() throws Exception
    {
        simpleValidationTest(1500);
    }

    /**
     * Test for CASSANDRA-5263
     * 1. Create N rows
     * 2. Run validation compaction
     * 3. Expect merkle tree with size 2^(log2(n))
     */
    public void simpleValidationTest(int n) throws Exception
    {
        Keyspace ks = Keyspace.open(keyspace);
        ColumnFamilyStore cfs = ks.getColumnFamilyStore(columnFamily);
        cfs.clearUnsafe();

        // disable compaction while flushing
        cfs.disableAutoCompaction();

        CompactionsTest.populate(keyspace, columnFamily, 0, n, 0); //ttl=3s

        cfs.forceBlockingFlush();
        assertEquals(1, cfs.getLiveSSTables().size());

        // wait enough to force single compaction
        TimeUnit.SECONDS.sleep(5);

        SSTableReader sstable = cfs.getLiveSSTables().iterator().next();
        UUID repairSessionId = UUIDGen.getTimeUUID();
        final RepairJobDesc desc = new RepairJobDesc(repairSessionId, UUIDGen.getTimeUUID(), cfs.keyspace.getName(),
                                               cfs.getTableName(), Collections.singletonList(new Range<>(sstable.first.getToken(),
                                                                                                                sstable.last.getToken())));

        ActiveRepairService.instance.registerParentRepairSession(repairSessionId, FBUtilities.getBroadcastAddress(),
                                                                 Collections.singletonList(cfs), desc.ranges, false, ActiveRepairService.UNREPAIRED_SSTABLE,
                                                                 PreviewKind.NONE);

        final CompletableFuture<Message> outgoingMessageSink = registerOutgoingMessageSink();
        Validator validator = new Validator(desc, FBUtilities.getBroadcastAddress(), 0, true, false, PreviewKind.NONE);
        CompactionManager.instance.submitValidation(cfs, validator);

        Message<ValidationComplete> message = outgoingMessageSink.get(TEST_TIMEOUT, TimeUnit.SECONDS);
        assertEquals(Verbs.REPAIR, message.group());
        assertEquals(Verbs.REPAIR.VALIDATION_COMPLETE, message.verb());
        ValidationComplete vc = message.payload();
        assertEquals(desc, vc.desc);
        assertTrue(vc.success());
        MerkleTrees trees = vc.trees;

        Iterator<Map.Entry<Range<Token>, MerkleTree>> iterator = trees.iterator();
        while (iterator.hasNext())
        {
            assertEquals(Math.pow(2, Math.ceil(Math.log(n) / Math.log(2))), iterator.next().getValue().size(), 0.0);
        }
        assertEquals(trees.rowCount(), n);
    }

    private CompletableFuture<Message> registerOutgoingMessageSink()
    {
        final CompletableFuture<Message> future = new CompletableFuture<>();
        MessagingService.instance().addInterceptor(new Interceptor()
        {
            public <M extends Message<?>> void intercept(M message, InterceptionContext<M> context)
            {
                if (context.isSending())
                    future.complete(message);
                context.drop(message);
            }
        });
        return future;
    }
}
